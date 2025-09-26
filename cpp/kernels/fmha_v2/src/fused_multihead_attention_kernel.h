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

#include <fmha/gmem_tile_o.h>
#include <fmha/gmem_tile_o_packed.h>
#include <fmha/gmem_tile_ps.h>
#include <fmha/gmem_tile_qkv.h>
#include <fmha/gmem_tile_qkv_packed.h>
#include <fmha/mask.h>
#include <fmha/smem_tile_o.h>
#include <fmha/smem_tile_qkv.h>
#include <fmha/smem_tile_v.h>
#include <fmha/softmax.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_cross_attention.h>

namespace fused_multihead_attention
{

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// The kernel implemented here reads the matrices K, Q and V of size (column-major):
//
// - K : EMBEDDING_SIZE * SEQUENCE_LENGTH (64 * 384 for BERT-Large),
// - Q : EMBEDDING_SIZE * SEQUENCE_LENGTH (64 * 384 for BERT-Large),
// - V : EMBEDDING_SIZE * SEQUENCE_LENGTH (64 * 384 for BERT-Large),
//
// It does the following operations:
//
// - P = norm * K^T * Q , where norm is the normalization term (a scalar),
// - S = Softmax(P) over the columns of P,
// - O = V * S.
//
// The intermediate matrices have the following sizes (column-major):
//
// - P : SEQUENCE_LENGTH * SEQUENCE_LENGTH (384 * 384 for BERT-Large),
// - O : EMBEDDING_SIZE  * SEQUENCE_LENGTH ( 64 * 384 for BERT-Large).
//
// To be able to hold the matrices on the SM we iterate over the SEQUENCE_LENGHT dimension of the
// O matrix (i.e. its columns). The matrices K and V are kept in registers on the SM whereas Q is
// read over the different iterations of the loop.
//
// To be able to operate entirely from registers (on Turing and Ampere) for the V * S product, we
// actually compute P^T = Q^T * K (remember that (AB)^T = B^T A^T).
//

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// U T I L S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FMHA_VERSION>
struct Single_cta
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Single_cta<1>
{

    // Ctor.
    template <typename Params>
    inline __device__ Single_cta(Params const& params, int bidb, int bidh, int bidn, int tidx)
        : bidb(bidb)
        , bidh(bidh)
        , bidn(bidn)
    {
        sum_s = params.b * params.s;
        actual_seqlen = params.s;
        bidx = bidb * params.h + bidh;
    }

    // Should we do an early exit? No.
    inline __device__ bool stop_early(int = 0) const
    {
        return false;
    }

    // The length of the sequence.
    int actual_seqlen;
    // The indices of the block (batch, head, linear index).
    int bidb, bidh, bidn, bidx;
    // The total number of tokens.
    int sum_s;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Single_cta<2>
{

    // Ctor for fmhca params. TODO: consolidate
    template <typename Params>
    inline __device__ Single_cta(Params const& params, int bidb, int bidh, int bidn, int tidx)
        : bidb(bidb)
        , bidh(bidh)
        , bidn(bidn)
        , num_heads(params.h)
    {
        sum_s = params.cu_seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s;
        bidx = sum_s * params.h + bidh;
    }

    // Ctor.
    inline __device__ Single_cta(
        bert::Fused_multihead_attention_params_v2 const& params, int bidb, int bidh, int bidn, int tidx)
        : bidb(bidb)
        , bidh(bidh)
        , bidn(bidn)
        , num_heads(params.h)
    {
        if (params.is_s_padded)
        {
            sum_s = params.s * bidb;
            // FIXME: might need s_kv here.
            sum_s_kv = params.s * bidb;
        }
        else
        {
            sum_s = params.cu_q_seqlens[bidb];
            sum_s_kv = params.cu_kv_seqlens[bidb];
        }
        actual_q_seqlen = params.cu_q_seqlens[bidb + 1] - params.cu_q_seqlens[bidb];
        actual_kv_seqlen
            = params.cu_kv_seqlens ? (params.cu_kv_seqlens[bidb + 1] - params.cu_kv_seqlens[bidb]) : actual_q_seqlen;
        actual_seqlen = actual_kv_seqlen;
        sum_mask_row = params.cu_mask_rows ? params.cu_mask_rows[bidb] : sum_s;
        bidx = sum_s * params.h + bidh;
    }

    // Skip empty sequences.
    inline __device__ bool stop_early(int loop = 0) const
    {
        return loop >= actual_q_seqlen;
    }

    // The length of the sequence.
    int actual_q_seqlen = 0;
    int actual_kv_seqlen = 0;
    // Keep for compatibility (it is the same as actual_kv_seqlen).
    int actual_seqlen = 0;
    // The total number of mask rows.
    int sum_mask_row;
    // The indices of the block (batch, head, linear index).
    int bidb, bidh, bidn, bidx;
    // The total number of q tokens.
    int sum_s;
    // The total number of kv tokens.
    int sum_s_kv;
    int num_heads;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int VERSION>
struct Multi_cta : public Single_cta<VERSION>
{

    // The base class.
    using Base = Single_cta<VERSION>;

    // Ctor.
    template <typename Params>
    inline __device__ Multi_cta(Params const& params, int bidb, int bidh, int bidn, int tidx)
        : Base(params, bidb, bidh, bidn, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Layout [Batch, Sequence Length]
template <int THREADS_PER_CTA, bool SEQUENCES_INTERLEAVED = false>
struct Block_info_padded
{

    template <typename Params>
    __device__ inline Block_info_padded(Params const& params, int const bidb, int const bidh, int const tidx)
        : bidb(bidb)
        , bidh(bidh)
        , bidn(0)
        , num_heads(params.h)
    {

        hidx = bidb * params.h + bidh;

        // The block index.
        sum_s = params.cu_seqlens[bidb];
        // actual_seqlen = params.seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s;
        bidx = sum_s * params.h + bidh;

        tidx_global = hidx * THREADS_PER_CTA + tidx;
    }

    __device__ inline bool stop_early() const
    {
        return actual_seqlen == 0;
    }

    template <int M_PER_ITER>
    __device__ inline int get_steps(int const begin) const
    {
        return ((actual_seqlen - begin) + M_PER_ITER - 1) / M_PER_ITER;
    }

    int actual_seqlen;
    int bidx;
    int sum_s;
    int bidh;
    int bidb;
    int bidn;
    int hidx;
    int num_heads;
    int tidx_global;
    int next_seq_offset_factor = 1;
};

// Layout [Sequence Length, Batch]
template <int THREADS_PER_CTA>
struct Block_info_padded<THREADS_PER_CTA, true>
{

    template <typename Params>
    __device__ inline Block_info_padded(Params const& params, int const bidb, int const bidh, int const tidx)
        : bidb(bidb)
        , bidh(bidh)
        , bidn(0)
        , num_heads(params.h)
    {

        hidx = bidb * params.h + bidh;

        // The block index.
        sum_s = bidb;
        // actual_seqlen = params.seqlens[bidb];
        actual_seqlen = params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
        bidx = sum_s * params.h + bidh;

        next_seq_offset_factor = params.b;

        tidx_global = hidx * THREADS_PER_CTA + tidx;
    }

    __device__ inline bool stop_early() const
    {
        return actual_seqlen == 0;
    }

    template <int M_PER_ITER>
    __device__ inline int get_steps(int const begin) const
    {
        return ((actual_seqlen - begin) + M_PER_ITER - 1) / M_PER_ITER;
    }

    int actual_seqlen;
    int bidx;
    int sum_s;
    int bidh;
    int bidb;
    int bidn;
    int hidx;
    int num_heads;
    int tidx_global;
    int next_seq_offset_factor;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
