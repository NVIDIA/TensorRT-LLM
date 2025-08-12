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

namespace fmha
{
namespace v2
{

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
    bool USE_LDGSTS_,
    // Are attention heads interleaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
    int NUM_MATS = 3>
struct Gmem_tile_tma_qkv
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

    // TMA DIMS, hard coded for now
    enum
    {
        TMA_DIMS = 3
    };

    // TMA DESC type, hard coded for now
    static constexpr fmha::cudaTmaDescType TMA_DESC_TYPE = fmha::cudaTmaDescType::TILED;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_tma_qkv(Params const& params, cudaTmaDesc const* p_desc, int qkv_offset,
        Block_info const& block_info, int tidx, int cta_row_offset = 0)
        // in PACKED_QKV, q_stride = k_stride = v_stride
        : params_qkv_stride_in_bytes_(params.q_stride_in_bytes)
        , actual_seqlen_(block_info.actual_seqlen)
        , qkv_ptr_(reinterpret_cast<char*>(params.qkv_ptr))
        , p_desc_(p_desc)
    {
        // Both MQA and GQA will use non HEADS_INTERLEAVED layout
        if (params.h_kv < params.h)
        {
            // QKV layout [b, s, [q_hd, k_h'd, v_h'd]]
            int const hi = block_info.bidh;
            int const hi_kv = block_info.bidh / (params.h / params.h_kv);
            if (qkv_offset == 0)
            { // Q tensor
                coord[0] = hi * params.d;
            }
            else if (qkv_offset == 1)
            { // K tensor
                coord[0] = params.h * params.d + hi_kv * params.d;
            }
            else if (qkv_offset == 2)
            { // V tensor
                coord[0] = params.h * params.d + params.h_kv * params.d + hi_kv * params.d;
            }
        }
        else
        {
            coord[0] = qkv_offset * params.d + block_info.bidh * params.d * 3;
        }
        // coord[1] = block_info.bidb * params.s; // should be params.s * batch_idx
        // coord[1] do not need to be adjusted per batch.
        // since the gmem_ptr in tma desc is set per batch and already adjusted.
        coord[1] = block_info.sum_s;
        coord[2] = 0;
    }

    // Store data to shared memory.
    template <typename Smem_tile>
    inline __device__ void commit(Smem_tile& smem_tile)
    {
    }

    // Load data from memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile& smem_tile)
    {
        smem_tile.template store<TMA_DIMS, TMA_DESC_TYPE>(p_desc_, coord);
    }

    // Store data to memory.
    inline __device__ void store(uint4 const (&data)[LDGS]) {}

    // Move the pointer to the next location.
    // only needed by matrix Q.
    inline __device__ void move()
    {
        // coord[1] is incremented by STEP size.
        coord[1] += ROWS;
    }

    // The stride between rows for the QKV matrice.
    int64_t params_qkv_stride_in_bytes_;
    // The pointer.
    char* qkv_ptr_;
    // The register to store predicates.
    uint32_t preds_[PRED_REGS];
    // The fetch registers.
    uint4 fetch_[LDGS];
    // Keep track of the row the thread is processing as we move the tile.
    int row_;
    // The sequence length.
    int actual_seqlen_;
    // tma descriptor
    cudaTmaDesc const* p_desc_;
    // coord use by TMA. For now hard code to 3D.
    int32_t coord[3];
};

} // namespace v2
} // namespace fmha
