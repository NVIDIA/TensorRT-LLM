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
#include "fmha/hopper/arrive_wait.h"
#include "fmha/hopper/smem_tile.h"
#include "fmha/utils.h"
#include <fmha/hopper/tma_descriptor.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/hopper/utils_tma.h>
#include <fused_multihead_attention_kernel.h>

namespace fmha
{
namespace ws
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
struct DMA
{

    // The shared struct.
    using Shared = typename Kernel_traits::Shared;
    // The kv buffer writer.
    using Circular_buffer_kv_writer = typename Kernel_traits::Circular_buffer_kv_writer;
    using Circular_buffer_v_scratch_reader = typename Kernel_traits::Circular_buffer_v_scratch_reader;
    using Circular_buffer_v_scratch_writer = typename Kernel_traits::Circular_buffer_v_scratch_writer;

    // The step size of Q loop.
    enum
    {
        STEP_Q = Kernel_traits::STEP_Q
    };

    // The step size of KV loop.
    enum
    {
        STEP_KV = Kernel_traits::STEP_KV
    };

    // The tile size of Q.
    enum
    {
        TILE_SIZE_Q = STEP_Q * Kernel_traits::D
    };

    // The tile size of Q after head_dimension split.
    enum
    {
        TILE_SIZE_Q_PER_D_GROUP = STEP_Q * Kernel_traits::D_PER_GROUP
    };

    // The tile size of K.
    enum
    {
        TILE_SIZE_K = STEP_KV * Kernel_traits::D
    };

    // The tile size of K after head_dimension split.
    enum
    {
        TILE_SIZE_K_PER_D_GROUP = STEP_KV * Kernel_traits::D_PER_GROUP
    };

    // The tile size of V.
    enum
    {
        TILE_SIZE_V = STEP_KV * Kernel_traits::DV
    };

    // The tile size of V after head_dimension split.
    enum
    {
        TILE_SIZE_V_PER_D_GROUP = TILE_SIZE_K_PER_D_GROUP
    };

    // Whether apply causal mask or not.
    enum
    {
        CAUSAL_MASK = Kernel_traits::CAUSAL_MASK
    };

    // Whether use custom mask input or not.
    enum
    {
        USE_CUSTOM_MASK = Kernel_traits::USE_CUSTOM_MASK
    };

    // Whether we skip those masked tiles when causal mask is enabled ?
    enum
    {
        SKIP_CAUSAL_MASK_TILES = CAUSAL_MASK && !USE_CUSTOM_MASK
    };

    // Whether we attend to the specific sliding window or chunk ?
    enum
    {
        SLIDING_OR_CHUNKED_ATTENTION = Kernel_traits::SLIDING_OR_CHUNKED_ATTENTION
    };

    // Is heads interleaved ?
    enum
    {
        HEADS_INTERLEAVED = Kernel_traits::HEADS_INTERLEAVED
    };

    // Named barrier for inter-warpgroup sync
    enum
    {
        SYNC_BARRIER = Kernel_traits::DMA_SYNC_BARRIER_ID
    };

    // The number of compute groups (currently fixed at 2).
    enum
    {
        NUM_COMPUTE_GROUPS = Kernel_traits::NUM_COMPUTE_GROUPS
    };

    // The tile scheduling mode: static (0), dynamic (1)
    enum
    {
        SCHEDULING_MODE = Kernel_traits::SCHEDULING_MODE
    };

    // Whether read from paged kv buffers or not.
    enum
    {
        PAGED_KV_INPUT = Kernel_traits::PAGED_KV_INPUT
    };

    // Whether the dma group transposes the v tile explicitly.
    enum
    {
        DMA_GROUP_TRANSPOSE_V = Kernel_traits::DMA_GROUP_TRANSPOSE_V
    };

    // How many threads get involved in the dma group.
    enum
    {
        NUM_THREADS_IN_DMA_GROUP = Kernel_traits::NUM_THREADS_IN_DMA_GROUP
    };

    // Transpose V
    // K is the sequence length dimension (128 for GMMA). The unroll factor is decided according to
    // empirical evidence so as to avoid register spill.
    enum
    {
        K_ = STEP_KV % 128 == 0 ? 128 : 64
    };

    static_assert(STEP_KV % K_ == 0);
    using Transposer = Transposer<typename Kernel_traits::Traits_o, typename Kernel_traits::Cta_tile_o, K_,
        (STEP_KV > 128 || SLIDING_OR_CHUNKED_ATTENTION) ? 1 : 2 /* UNROLL */>;

    struct Device
    {
        // Only the warpgroup leader initiates mbarriers & TMA operations.
        uint32_t elect_one_;
        // The sum_s for q.
        int sum_s_q_;
        // The sum_s for kv.
        int sum_s_kv_;
        // multi_query_attention (multiple heads share the same key/value).
        bool multi_query_attention_;
        // Tile id for q tile scheduling
        uint32_t tile_id_;

        inline __device__ Device(uint32_t elect_one)
            : elect_one_(elect_one)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////

        // Compute the kv tile idx start (inclusive) and end (exclusive).
        static inline __device__ std::pair<int, int> compute_kv_tile_idx(
            bert::Fused_multihead_attention_params_v2 const& params, int q_step_offset, int q_step_end, int kv_steps)
        {

            // The default kv_idx_start and kv_idx_end (exclusive).
            int kv_idx_start = 0;
            int kv_idx_end = kv_steps;

            // Is the chunked_attention used ?
            bool is_chunked_attention = params.log2_chunked_attention_size > 0;

            // Skip initial kv tiles due to sliding_window_size
            if (SLIDING_OR_CHUNKED_ATTENTION)
            {
                // The kv_offset_start.
                int kv_offset_start = is_chunked_attention
                    ? ((q_step_offset >> params.log2_chunked_attention_size) << params.log2_chunked_attention_size)
                    : max(0, q_step_offset - params.sliding_window_size);
                kv_idx_start = kv_offset_start / STEP_KV;
            }

            // Early stop when causal mask is enabled.
            if (SKIP_CAUSAL_MASK_TILES)
            {
                kv_idx_end = (q_step_end + STEP_KV - 1) / STEP_KV;
            }

            return std::make_pair(kv_idx_start, kv_idx_end);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////

        // Packed contiguous QKV input.
        inline __device__ void run_packed_qkv(bert::Fused_multihead_attention_params_v2 const& params, Shared* shared)
        {
            // DMA.
            int local_wid = (threadIdx.x / 32) % 4;
            int tiw = threadIdx.x % 32;
            uint32_t smem_tile_id = __cvta_generic_to_shared(&shared->tile_id);

            if (SCHEDULING_MODE == 0)
            {
                tile_id_ = blockIdx.y;
            }
            else
            {
                get_next_tile_id(local_wid, tiw, smem_tile_id, params.tile_id_counter_ptr);
            }

            auto cbw0 = shared->tma_q_tracker[0].createWriter();
            auto cbw1 = shared->tma_q_tracker[1].createWriter();
            Circular_buffer_kv_writer cbw_k = shared->tma_k_tracker.createWriter();
            Circular_buffer_kv_writer cbw_v = shared->tma_v_tracker.createWriter();
            Circular_buffer_v_scratch_reader cbr_v_scratch = shared->tma_v_scratch_tracker.createReader();
            Circular_buffer_v_scratch_writer cbw_v_scratch = shared->tma_v_scratch_tracker.createWriter();
            auto headinfo_tracker0 = shared->head_info_tracker[0].createWriter();
            auto headinfo_tracker1 = shared->head_info_tracker[1].createWriter();

            // When compiled for TRT-LLLM (heads_interleaved = false), this flag won't make a difference.
            multi_query_attention_ = params.h_kv < params.h;

            while (tile_id_ < params.num_tiles)
            {
                // If we do bidh = next_head % h, we'd guarantee b to be spread across CTAs.

                int bidb, tmp, bidh, q_step_offset, q_steps;

                if (SCHEDULING_MODE == 0)
                {
                    bidh = tile_id_ % params.h;
                    bidb = tile_id_ / params.h;
                }
                else
                {
                    // Balanced dynamic scheduling
                    if (CAUSAL_MASK && !SLIDING_OR_CHUNKED_ATTENTION && params.use_balanced_scheduling)
                    {
                        q_step_offset
                            = (params.num_tiles_per_head - 1 - tile_id_ / (params.b * params.h)) * NUM_COMPUTE_GROUPS;
                        tmp = tile_id_ % (params.b * params.h);
                        bidh = tmp / params.b;
                        bidb = tmp % params.b;
                        q_steps = NUM_COMPUTE_GROUPS;
                    }
                    else
                    { // Unbalanced dynamic scheduling
                        bidb = tile_id_ / (params.h * params.num_tiles_per_head);
                        tmp = tile_id_ % (params.h * params.num_tiles_per_head);
                        bidh = tmp / params.num_tiles_per_head;
                        q_step_offset = tmp % params.num_tiles_per_head * NUM_COMPUTE_GROUPS;
                        q_steps = NUM_COMPUTE_GROUPS;
                    }
                }

                cudaTmaDesc const* desc_q = &params.tma_desc_q;
                cudaTmaDesc const* desc_kv = &params.tma_desc_kv;
                cudaTmaDesc const* desc_v = &params.tma_desc_v;
                int actual_seqlen;
                if (params.is_s_padded)
                {
                    sum_s_q_ = bidb * params.s;
                    actual_seqlen = params.cu_q_seqlens[bidb + 1] - params.cu_q_seqlens[bidb];
                }
                else
                {
                    sum_s_q_ = params.cu_q_seqlens[bidb];
                    actual_seqlen = params.cu_q_seqlens[bidb + 1] - sum_s_q_;
                }

                // The cumulative packed_mask seqlens.
                // Each sequence length in the batch has to be padded to multiple of 128.
                int sum_mask_s = params.cu_mask_rows[bidb];

                if (SCHEDULING_MODE == 0)
                {
                    // split work across M
                    q_steps = (actual_seqlen + STEP_Q - 1) / STEP_Q;

                    // Q_steps may be distributed to multiple blocks to increase the occupacy
                    // when b*h is small.
                    // The number of q_steps needs to be multiple of 2.
                    q_steps = (q_steps + gridDim.x - 1) / gridDim.x;
                    q_steps += (q_steps & 1);
                    // The last block may process fewer q_steps.
                    q_step_offset = q_steps * blockIdx.x;
                }

                int q_tile_offset = q_step_offset * STEP_Q;
                if (q_tile_offset >= actual_seqlen)
                {
                    if (SCHEDULING_MODE == 0)
                    {
                        tile_id_ += gridDim.y;
                    }
                    else
                    {
                        get_next_tile_id(local_wid, tiw, smem_tile_id, params.tile_id_counter_ptr);
                    }
                    continue;
                }

                // Split work across N.
                int const kv_steps = (actual_seqlen + STEP_KV - 1) / STEP_KV;

                for (int q_step_idx = 0; q_step_idx < q_steps; q_step_idx += 2)
                {
                    load_q(bidh, q_step_idx + 0 + q_step_offset, desc_q, shared->smem_q[0], cbw0);
                    load_q(bidh, q_step_idx + 1 + q_step_offset, desc_q, shared->smem_q[1], cbw1);

                    // Q step bound is 2 tiles away at this moment because of 2x1 math warpgroup
                    int const q_step_end = (q_step_idx + q_step_offset + 2) * STEP_Q - 1;

                    // The kv tile idx range for this q step.
                    auto const [kv_idx_start, kv_idx_end]
                        = compute_kv_tile_idx(params, (q_step_idx + q_step_offset) * STEP_Q, q_step_end, kv_steps);

                    // Iterate over the kv tiles for this q step.
                    for (int kv_step_idx = kv_idx_start; kv_step_idx < kv_idx_end; kv_step_idx++)
                    {
                        int bar_id = load_kv(bidh, params.h, params.h_kv, kv_step_idx, desc_kv, desc_v, shared, cbw_k,
                            cbw_v, cbw_v_scratch, cbr_v_scratch);

                        // Opportunistically hide headinfo in the shadow of UTMALDGs of the QKV tensor
                        if (q_step_idx == 0 && kv_step_idx == kv_idx_start)
                        {
                            // Send head info.
                            typename Shared::Head_info info{q_steps,
                                // q, and kv have the same length.
                                q_tile_offset, USE_CUSTOM_MASK ? sum_mask_s : q_tile_offset, kv_steps,
                                // q, and kv have the same length.
                                actual_seqlen, actual_seqlen, sum_s_q_ * params.h + bidh, bidh, bidb};
                            // NOTE: The need for the sync after consumer bar wait is to avoid a deadlock hazard
                            // when DMA thread 0 is ahead of other DMA threads. For example:
                            // DMA thread 0 have finished consumer bar wait phase 0 and producer bar arrive phase 0, and
                            // then MMA warps have finished producer bar wait phase 0 and consumer bar arrive phase 1.
                            // At this time other DMA threads start consumer bar wait phase 0. It will never become
                            // ready. DMA warps then fail to continue to the next loop.
                            //
                            // It is the same consideration for the sync after tmaReserve in load_q and load_kv
                            // implementation below.
                            headinfo_tracker0.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                                elect_one_, info);
                            headinfo_tracker1.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                                elect_one_, info);
                        }

                        if constexpr (DMA_GROUP_TRANSPOSE_V)
                        {
                            transpose_v_tile(bar_id, shared, cbw_v, cbr_v_scratch);
                        }
                    } // kv
                }     // q

                if (SCHEDULING_MODE == 0)
                {
                    tile_id_ += gridDim.y;
                }
                else
                {
                    get_next_tile_id(local_wid, tiw, smem_tile_id, params.tile_id_counter_ptr);
                }
            } // gridDim.y
            // Signal compute groups to break.
            headinfo_tracker0.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                elect_one_, {-1, -1, -1, -1, -1, -1, -1, -1});
            headinfo_tracker1.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                elect_one_, {-1, -1, -1, -1, -1, -1, -1, -1});
        }

        // Calculate the start tile idx.
        inline __device__ int remap_kv_tile_idx(
            int kv_tile_idx, int num_kv_cache_tiles, int past_kv_length, int sliding_window_size)
        {

            // The remapped kv tile idx.
            int remapped_kv_tile_idx = kv_tile_idx;
            // This will be removed later as the remapping will be handled by the kvCacheManger in TRTLLM.
#ifdef GENERATE_CUBIN
            // Sliding window attention + chunked context needs special handling.
            if constexpr (SLIDING_OR_CHUNKED_ATTENTION)
            {
                // For chunked context (i.e. separate q and kv layout), the kv cache might be
                // overwritten after last chunk is processed.
                // To deal with this issue, the new tokens' kv will be appended to the kv cache first,
                // and overwrite the kv cache after FMHA is done.
                // The kv input layout is like: [cyclic kv cache] + [new tokens' kv].
                // There are two possible cases:
                // 1. The kv cache hasn't been overwritten while processing previous chunks, so we can
                //    take it normally, where we have full kv cache.
                // 2. The kv cache has been overwritten while processing previous chunks. we need to
                //    mask out the tokens in the kv cache based on the sliding window size. It needs
                //    to track the last kv cache token's position in a circular way.

                // Remap the kv tile index when kv cache has been overwritten in a circular way.
                if (past_kv_length > sliding_window_size)
                {
                    // Map the kv tile index to the new tokens' kv.
                    if (kv_tile_idx * STEP_KV >= past_kv_length)
                    {
                        remapped_kv_tile_idx
                            = num_kv_cache_tiles + int((kv_tile_idx * STEP_KV - past_kv_length) / STEP_KV);
                    }
                    else
                    {
                        // Map the kv tile index to the cyclic kv cache.
                        remapped_kv_tile_idx = kv_tile_idx % num_kv_cache_tiles;
                    }
                }
            }
#endif
            // Return the remapped kv tile idx.
            return remapped_kv_tile_idx;
        }

        // Support contiguous Q + contiguous/paged KV separate cache.
        inline __device__ void run_separate_q_and_kv(
            bert::Fused_multihead_attention_params_v2 const& params, Shared* shared)
        {
            // DMA.
            int local_wid = (threadIdx.x / 32) % 4;
            int tiw = threadIdx.x % 32;
            uint32_t smem_tile_id = __cvta_generic_to_shared(&shared->tile_id);

            if (SCHEDULING_MODE == 0)
            {
                tile_id_ = blockIdx.y;
            }
            else
            {
                get_next_tile_id(local_wid, tiw, smem_tile_id, params.tile_id_counter_ptr);
            }

            auto cbw0 = shared->tma_q_tracker[0].createWriter();
            auto cbw1 = shared->tma_q_tracker[1].createWriter();
            Circular_buffer_kv_writer cbw_k = shared->tma_k_tracker.createWriter();
            Circular_buffer_kv_writer cbw_v = shared->tma_v_tracker.createWriter();
            Circular_buffer_v_scratch_reader cbr_v_scratch = shared->tma_v_scratch_tracker.createReader();
            Circular_buffer_v_scratch_writer cbw_v_scratch = shared->tma_v_scratch_tracker.createWriter();
            auto headinfo_tracker0 = shared->head_info_tracker[0].createWriter();
            auto headinfo_tracker1 = shared->head_info_tracker[1].createWriter();

            while (tile_id_ < params.num_tiles)
            {
                // If we do bidh = next_head % h, we'd guarantee b to be spread across CTAs.

                int bidb, tmp, bidh, local_q_tile_offset, q_steps;

                if (SCHEDULING_MODE == 0)
                {
                    bidh = tile_id_ % params.h;
                    bidb = tile_id_ / params.h;
                }
                else if (SCHEDULING_MODE == 1)
                {
                    bidb = tile_id_ / (params.h * params.num_tiles_per_head);
                    tmp = tile_id_ % (params.h * params.num_tiles_per_head);
                    bidh = tmp / params.num_tiles_per_head;
                    local_q_tile_offset = (tmp % params.num_tiles_per_head) * NUM_COMPUTE_GROUPS * STEP_Q;
                    q_steps = NUM_COMPUTE_GROUPS;
                }
                else
                { // SCHEDULING_MODE == 2
                    local_q_tile_offset = (params.num_tiles_per_head - 1 - tile_id_ / (params.b * params.h))
                        * NUM_COMPUTE_GROUPS * STEP_Q;
                    tmp = tile_id_ % (params.b * params.h);
                    bidh = tmp / params.b;
                    bidb = tmp % params.b;
                    q_steps = NUM_COMPUTE_GROUPS;
                }
                int bidh_kv = bidh / params.h_q_per_kv;

                // Sequence length parameters.
                // Take chunked attention (q, and kv may have difference sequence length) into consideration.
                sum_s_q_ = params.is_s_padded ? bidb * params.s : params.cu_q_seqlens[bidb];
                sum_s_kv_ = params.is_s_padded ? bidb * params.s : params.cu_kv_seqlens[bidb];
                int actual_q_seqlen = params.cu_q_seqlens[bidb + 1] - params.cu_q_seqlens[bidb];
                int actual_kv_seqlen = params.cu_kv_seqlens[bidb + 1] - params.cu_kv_seqlens[bidb];
                int past_kv_length = actual_kv_seqlen - actual_q_seqlen;

                // The cumulative packed_mask seqlens.
                // Each sequence length in the batch has to be padded to multiple of 128.
                int sum_mask_s = params.cu_mask_rows[bidb];

                // Prepare the tma descriptors.
                cudaTmaDesc const* desc_q = &params.tma_desc_q;
                int32_t const* paged_block_offsets
                    = params.paged_kv_cache.mBlockOffsets + bidb * 2 * params.paged_kv_cache.mMaxBlocksPerSeq;
                cudaTmaDesc const* desc_kv = &params.tma_desc_kv;
                // If a separate v_stride_in_bytes is set, we have to use separate tma_desc_v,
                // otherwise share with tma_desc_kv.
                // This is for the compatibility that TensorRT-LLM needs no modification if padding V to 192.
#ifndef GENERATE_CUBIN
                cudaTmaDesc const* desc_v
                    = (params.v_stride_in_bytes == 0 || params.v_stride_in_bytes == params.kv_stride_in_bytes)
                    ? desc_kv
                    : &params.tma_desc_v;
#else
                cudaTmaDesc const* desc_v = desc_kv;
#endif
                if (SCHEDULING_MODE == 0)
                {
                    // split work across M
                    q_steps = (actual_q_seqlen + STEP_Q - 1) / STEP_Q;

                    // Q_steps may be distributed to multiple blocks to increase the occupacy
                    // when b*h is small.
                    // The number of q_steps needs to be multiple of 2.
                    q_steps = (q_steps + gridDim.x - 1) / gridDim.x;
                    q_steps += (q_steps & 1);
                    local_q_tile_offset = q_steps * blockIdx.x * STEP_Q;
                }

                // The last block may process fewer q_steps.
                if (local_q_tile_offset >= actual_q_seqlen)
                {
                    if (SCHEDULING_MODE == 0)
                    {
                        tile_id_ += gridDim.y;
                    }
                    else
                    {
                        get_next_tile_id(local_wid, tiw, smem_tile_id, params.tile_id_counter_ptr);
                    }
                    continue;
                }

                // The global q tile offset which includes the past kv cache.
                int q_tile_offset = local_q_tile_offset + past_kv_length;
                // Split work across N.
                int const kv_steps = (actual_kv_seqlen + STEP_KV - 1) / STEP_KV;
                // Page KV: number of valid kv blocks (others might be nullptr).
                int const num_valid_kv_blocks = (actual_kv_seqlen + params.paged_kv_cache.mTokensPerBlock - 1)
                    >> params.paged_kv_cache.mTokensPerBlockLog2;

                for (int q_step_idx = 0; q_step_idx < q_steps; q_step_idx += 2)
                {
                    load_separate_q(bidh, q_step_idx * STEP_Q + local_q_tile_offset, desc_q, shared->smem_q[0], cbw0);
                    load_separate_q(
                        bidh, (q_step_idx + 1) * STEP_Q + local_q_tile_offset, desc_q, shared->smem_q[1], cbw1);

                    // Q step end is 2 tiles away at this moment because of 2x1 math warpgroup
                    int const q_step_end = (q_step_idx + 2) * STEP_Q - 1 + q_tile_offset;

                    // The kv tile idx range for this q step.
                    auto const [kv_idx_start, kv_idx_end]
                        = compute_kv_tile_idx(params, q_step_idx * STEP_Q + q_tile_offset, q_step_end, kv_steps);

                    // Iterate over the kv tiles for this q step.
                    for (int kv_step_idx = kv_idx_start; kv_step_idx < kv_idx_end; kv_step_idx++)
                    {
                        // Remap the kv tile idx if sliding window attention is enabled.
                        // Sliding_window_size should be multiple of STEP_KV.
                        int remapped_kv_step_idx = remap_kv_tile_idx(kv_step_idx, params.sliding_window_size / STEP_KV,
                            past_kv_length, params.sliding_window_size);
                        // The barrier id.
                        int bar_id;
                        // Load paged kv input.
                        if constexpr (PAGED_KV_INPUT)
                        {
                            bar_id = load_paged_kv(bidh_kv, remapped_kv_step_idx * STEP_KV, num_valid_kv_blocks,
                                params.paged_kv_cache.mTokensPerBlockLog2, params.blocks_per_tma_load,
                                params.blocks_per_tma_load_log2, params.paged_kv_cache.mMaxBlocksPerSeq,
                                paged_block_offsets, desc_kv, desc_v, shared, cbw_k, cbw_v, cbw_v_scratch,
                                cbr_v_scratch);
                        }
                        else
                        {
                            bar_id = load_contiguous_kv(bidh, params.h, params.h_kv, remapped_kv_step_idx, desc_kv,
                                shared, cbw_k, cbw_v, cbw_v_scratch, cbr_v_scratch);
                        }

                        // Opportunistically hide headinfo in the shadow of UTMALDGs of the QKV tensor
                        if (q_step_idx == 0 && kv_step_idx == kv_idx_start)
                        {
                            // Send head info.
                            typename Shared::Head_info info{q_steps, local_q_tile_offset,
                                USE_CUSTOM_MASK ? sum_mask_s : q_tile_offset, kv_steps, actual_q_seqlen,
                                actual_kv_seqlen, sum_s_q_ * params.h + bidh, bidh, bidb};
                            headinfo_tracker0.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                                elect_one_, info);
                            headinfo_tracker1.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                                elect_one_, info);
                        }
                        if constexpr (DMA_GROUP_TRANSPOSE_V)
                        {
                            transpose_v_tile(bar_id, shared, cbw_v, cbr_v_scratch);
                        }
                    } // kv
                }     // q

                if (SCHEDULING_MODE == 0)
                {
                    tile_id_ += gridDim.y;
                }
                else
                {
                    get_next_tile_id(local_wid, tiw, smem_tile_id, params.tile_id_counter_ptr);
                }
            } // gridDim.y

            // Signal compute groups to break.
            headinfo_tracker0.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                elect_one_, {-1, -1, -1, -1, -1, -1, -1, -1});
            headinfo_tracker1.template push_with_sync<SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP>(
                elect_one_, {-1, -1, -1, -1, -1, -1, -1, -1});
        }

        // Load q tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename Smem_q>
        inline __device__ void load_q(
            int bidh, int q_step_idx, cudaTmaDesc const* desc_q, Smem_q& smem_q, BufferWriter& cbw)
        {

            int barrier_id = cbw.tmaReserve(elect_one_, TILE_SIZE_Q * Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

            // coordinates: d, 3, h, s
            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
            int32_t const q_coord_dim1 = !HEADS_INTERLEAVED || multi_query_attention_ ? bidh : 0;
            int32_t const q_coord_dim2 = !HEADS_INTERLEAVED || multi_query_attention_ ? 0 : bidh;
#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, q_coord_dim1, q_coord_dim2, sum_s_q_ + q_step_idx * STEP_Q};
                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_q,
                    __cvta_generic_to_shared(&smem_q[barrier_id * TILE_SIZE_Q + di * TILE_SIZE_Q_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw.barrier_ptr(barrier_id)), coords, elect_one_);
            }
        }

        // Load q tiles from gmem to smem by TMA.
        // Only has q tiles in this buffer, kv tiles are read from paged kv buffers.
        template <typename BufferWriter, typename Smem_q>
        inline __device__ void load_separate_q(
            int bidh, int q_tile_start_offset, cudaTmaDesc const* desc_q, Smem_q& smem_q, BufferWriter& cbw)
        {

            int barrier_id = cbw.tmaReserve(elect_one_, TILE_SIZE_Q * Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

// coordinates: d, h, 1, s
// split D into multiple groups in order to satisfy the TMA 128B sizzle mode
#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const coords[4] = {di * Kernel_traits::D_PER_GROUP, bidh, 0, sum_s_q_ + q_tile_start_offset};
                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_q,
                    __cvta_generic_to_shared(&smem_q[barrier_id * TILE_SIZE_Q + di * TILE_SIZE_Q_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw.barrier_ptr(barrier_id)), coords, elect_one_);
            }
        }

        // Load k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter>
        inline __device__ void load_kv_impl(int bidh, int h, int h_kv, int kv_step_idx, cudaTmaDesc const* desc_kv,
            cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v)
        {

            int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);

            int v_barrier_id = cbw_v.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

            // Coordinates:
            // [d, 3, h, s] for head_interleaved, otherwise [d, h, 3, s]
            // for multi_query attention, it will be [d, h + 2, 1, s]
            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
            int32_t const k_coord_dim1 = HEADS_INTERLEAVED ? 1 : bidh;
            int32_t const k_coord_dim2 = HEADS_INTERLEAVED ? bidh : 1;

#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const k_coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, multi_query_attention_ ? h + bidh / (h / h_kv) : k_coord_dim1,
                        multi_query_attention_ ? 0 : k_coord_dim2, sum_s_q_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                    __cvta_generic_to_shared(
                        &shared->smem_k[k_barrier_id * TILE_SIZE_K + di * TILE_SIZE_K_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
            }
#pragma unroll
            for (int di = 0; di < Kernel_traits::DV_GROUPS; ++di)
            {
                int32_t const v_coords[4] = {di * Kernel_traits::D_PER_GROUP,
                    multi_query_attention_ ? bidh / (h / h_kv) : bidh, 0, sum_s_q_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_v,
                    __cvta_generic_to_shared(
                        &shared->smem_v[v_barrier_id * TILE_SIZE_V + di * TILE_SIZE_V_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_v.barrier_ptr(v_barrier_id)), v_coords, elect_one_);
            }
        }

        // Load contiguous kv tiles [B, S, 2, H, D] from gmem to smem by TMA.
        template <typename BufferWriter>
        inline __device__ void load_contiguous_kv_impl(int bidh, int h, int h_kv, int kv_step_idx,
            cudaTmaDesc const* desc_kv, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v)
        {

            int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);

            int v_barrier_id = cbw_v.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const k_coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, bidh / (h / h_kv), 0, sum_s_kv_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                    __cvta_generic_to_shared(
                        &shared->smem_k[k_barrier_id * TILE_SIZE_K + di * TILE_SIZE_K_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);

                int32_t const v_coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, bidh / (h / h_kv), 1, sum_s_kv_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                    __cvta_generic_to_shared(
                        &shared->smem_v[v_barrier_id * TILE_SIZE_V + di * TILE_SIZE_V_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_v.barrier_ptr(v_barrier_id)), v_coords, elect_one_);
            }
        }

        // Load k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter>
        inline __device__ void load_paged_kv_impl(int bidh, int kv_tile_start_offset, int num_valid_kv_blocks,
            int tokens_per_block_log2, int blocks_per_tma_load, int blocks_per_tma_load_log2,
            int max_blocks_per_sequence, int32_t const* paged_block_offsets, cudaTmaDesc const* desc_kv,
            cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v)
        {

            int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);

            int v_barrier_id = cbw_v.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

            // Paged KV cache block idx.
            int paged_kv_block_idx = kv_tile_start_offset >> tokens_per_block_log2;
            int kv_offset_in_block = kv_tile_start_offset & ((1 << tokens_per_block_log2) - 1);

            // coordinates: d, s, h, 1
            int const tile_size_k_per_block = TILE_SIZE_K_PER_D_GROUP >> blocks_per_tma_load_log2;
            static_assert(
                TILE_SIZE_V_PER_D_GROUP == TILE_SIZE_K_PER_D_GROUP, "KV tile should have the same tensor size.");
            for (int bi = 0; bi < blocks_per_tma_load; ++bi)
            {
                int const bounded_block_idx = min(num_valid_kv_blocks - 1, paged_kv_block_idx + bi);

                int32_t const k_paged_block_offset = paged_block_offsets[bounded_block_idx];
                int32_t const v_paged_block_offset = paged_block_offsets[max_blocks_per_sequence + bounded_block_idx];

#pragma unroll
                for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
                {
                    int32_t const k_coords[4]
                        = {di * Kernel_traits::D_PER_GROUP, kv_offset_in_block, bidh, k_paged_block_offset};

                    fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                        __cvta_generic_to_shared(&shared->smem_k[k_barrier_id * TILE_SIZE_K
                            + di * TILE_SIZE_K_PER_D_GROUP + bi * tile_size_k_per_block]),
                        __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
                }
#pragma unroll
                for (int di = 0; di < Kernel_traits::DV_GROUPS; ++di)
                {
                    int32_t const v_coords[4]
                        = {di * Kernel_traits::D_PER_GROUP, kv_offset_in_block, bidh, v_paged_block_offset};

                    fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_v,
                        __cvta_generic_to_shared(&shared->smem_v[v_barrier_id * TILE_SIZE_V
                            + di * TILE_SIZE_V_PER_D_GROUP + bi * tile_size_k_per_block]),
                        __cvta_generic_to_shared(cbw_v.barrier_ptr(v_barrier_id)), v_coords, elect_one_);
                }
            }
        }

        template <typename BufferWriter, typename BufferReaderScratch>
        // Transpose v tile explicitly as QGMMA doesn't support it.
        inline __device__ void transpose_v_tile(
            int v_scratch_barrier_id, Shared* shared, BufferWriter& cbw_v, BufferReaderScratch& cbr_v_scratch)
        {
            static_assert(NUM_THREADS_IN_DMA_GROUP == 128, "");
            Transposer transposer(threadIdx.x % NUM_THREADS_IN_DMA_GROUP);

            // Src buffer available
            int ready = cbr_v_scratch.peek();
            if (!ready)
            {
                cbr_v_scratch.wait();
            }
            uint32_t smem_v_src = __cvta_generic_to_shared(&shared->smem_v_scratch[v_scratch_barrier_id]);

            // Dst buffer available
            int v_barrier_id = cbw_v.threadReserve();
            uint32_t smem_v_dst = __cvta_generic_to_shared(&shared->smem_v[v_barrier_id * TILE_SIZE_V]);

// Explicitly transpose the v buffer in smem for fp8.

// The transposer currently has support of the following tile sizes:
//   - D=32, S (or KV_STEP)=128
//   - D=64, S (or KV_STEP)=64, 128
//   - D=128, S (or KV_STEP)=64, 128
// In addition, the transposer can only work with contiguous chunk of SMEM.
//
// For example, if V tile size is D=256 S=256, we can divide the TMA load of the V tile
// (SxD) into 2x2 chunks of size 128x128. This way, when tiles (0, 0), (0, 1) are transposed,
// either the load and the store of the data can be performed in a contiguous memory.
//
// Keep in mind in order to match GMMA requirement, we need to store the transposed tiles
// along D dim first then S dim. Leading dimension S after the transpose is at most 128B.
//
// Logical:
//         D  -  D I M  (contiguous)
//
//           128            128          S
//     <------------> <------------>     -
//     s, d = (0, 0) | s, d = (0, 1)     D
//     ------------------------------    I
//     s, d = (1, 0) | s, d = (1, 1)     M
//
// In SMEM:
//                             D  -  D I M
//
//           128            128             128            128         S
//     <------------> <-------------> <-------------> <------------>   -
//     s, d = (0, 0) | s, d = (0, 1) | s, d = (1, 0) | s, d = (1, 1)   D  (contiguous)
//                                                                     I
//                                                                     M
//
#pragma unroll
            for (int kgroup_idx = 0; kgroup_idx < Kernel_traits::BMM2_K_GROUPS; kgroup_idx++)
            {
#pragma unroll
                for (int dgroup_idx = 0; dgroup_idx < Kernel_traits::D_GROUPS; dgroup_idx++)
                {
                    // Src smem block is k first then d
                    uint32_t src_offset = (kgroup_idx * Kernel_traits::BMM2_K_PER_GROUP * Kernel_traits::D_PER_GROUP
                                              + dgroup_idx * Kernel_traits::D_PER_GROUP * Kernel_traits::STEP_KV)
                        * Kernel_traits::ELEMENT_BYTES;

                    // Dst smem block is d first then k
                    uint32_t dst_offset = (dgroup_idx * Kernel_traits::BMM2_K_PER_GROUP * Kernel_traits::D_PER_GROUP
                                              + kgroup_idx * Kernel_traits::BMM2_K_PER_GROUP * Kernel_traits::D)
                        * Kernel_traits::ELEMENT_BYTES;

                    transposer.template transpose_<false>(smem_v_src + src_offset, smem_v_dst + dst_offset);
                }
            }

            fence_view_async_shared();                                  // Commit STSM
            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP); // Sync before signaling
            cbw_v.threadCommit(elect_one_, v_barrier_id);               // Signal readiness
            cbr_v_scratch.pop(elect_one_);                              // Advance to next phase
        }

        // Load k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch, typename BufferReaderScratch>
        inline __device__ int load_kv_transpose_v_impl(int bidh, int h, int h_kv, int kv_step_idx,
            cudaTmaDesc const* desc_kv, cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k,
            BufferWriter& cbw_v, BufferWriterScratch& cbw_v_scratch, BufferReaderScratch& cbr_v_scratch)
        {
            int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

            // Coordinates:
            // [d, 3, h, s] for head_interleaved, otherwise [d, h, 3, s]
            // for multi_query attention, it will be [d, h + 2, 1, s]
            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
            int32_t const k_coord_dim1 = HEADS_INTERLEAVED ? 1 : bidh;
            int32_t const k_coord_dim2 = HEADS_INTERLEAVED ? bidh : 1;

#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const k_coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, multi_query_attention_ ? h + bidh / (h / h_kv) : k_coord_dim1,
                        multi_query_attention_ ? 0 : k_coord_dim2, sum_s_q_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                    __cvta_generic_to_shared(
                        &shared->smem_k[k_barrier_id * TILE_SIZE_K + di * TILE_SIZE_K_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
            }

            int v_scratch_barrier_id
                = cbw_v_scratch.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);

#pragma unroll
            for (int di = 0; di < Kernel_traits::DV_GROUPS; ++di)
            {
                int32_t const v_coords[4] = {di * Kernel_traits::D_PER_GROUP,
                    multi_query_attention_ ? bidh / (h / h_kv) : bidh, 0, sum_s_q_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_v,
                    __cvta_generic_to_shared(
                        &shared->smem_v_scratch[v_scratch_barrier_id * TILE_SIZE_V + di * TILE_SIZE_V_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_v_scratch.barrier_ptr(v_scratch_barrier_id)), v_coords, elect_one_);
            }

            // Do we really need this as we only have one buffer ?
            return v_scratch_barrier_id;
        }

        // Load contiguous kv tiles [B, S, 2, H, D] from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch, typename BufferReaderScratch>
        inline __device__ int load_contiguous_kv_transpose_v_impl(int bidh, int h, int h_kv, int kv_step_idx,
            cudaTmaDesc const* desc_kv, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v,
            BufferWriterScratch& cbw_v_scratch, BufferReaderScratch& cbr_v_scratch)
        {
            int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const k_coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, bidh / (h / h_kv), 0, sum_s_kv_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                    __cvta_generic_to_shared(
                        &shared->smem_k[k_barrier_id * TILE_SIZE_K + di * TILE_SIZE_K_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
            }

            int v_scratch_barrier_id
                = cbw_v_scratch.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);

#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                int32_t const v_coords[4]
                    = {di * Kernel_traits::D_PER_GROUP, bidh / (h / h_kv), 1, sum_s_kv_ + kv_step_idx * STEP_KV};

                fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                    __cvta_generic_to_shared(
                        &shared->smem_v_scratch[v_scratch_barrier_id * TILE_SIZE_V + di * TILE_SIZE_V_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_v_scratch.barrier_ptr(v_scratch_barrier_id)), v_coords, elect_one_);
            }

            // Do we really need this as we only have one buffer ?
            return v_scratch_barrier_id;
        }

        // Load paged k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch, typename BufferReaderScratch>
        inline __device__ int load_paged_kv_transpose_v_impl(int bidh, int kv_tile_start_offset,
            int num_valid_kv_blocks, int tokens_per_block_log2, int blocks_per_tma_load, int blocks_per_tma_load_log2,
            int max_blocks_per_sequence, int32_t const* paged_block_offsets, cudaTmaDesc const* desc_kv, Shared* shared,
            BufferWriter& cbw_k, BufferWriter& cbw_v, BufferWriterScratch& cbw_v_scratch,
            BufferReaderScratch& cbr_v_scratch)
        {
            int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);

            int v_scratch_barrier_id
                = cbw_v_scratch.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

            // Paged KV cache block idx.
            int paged_kv_block_idx = kv_tile_start_offset >> tokens_per_block_log2;
            int kv_offset_in_block = kv_tile_start_offset & ((1 << tokens_per_block_log2) - 1);

            // coordinates: d, s, h, 1
            int const tile_size_k_per_block = TILE_SIZE_K_PER_D_GROUP >> blocks_per_tma_load_log2;
            static_assert(
                TILE_SIZE_V_PER_D_GROUP == TILE_SIZE_K_PER_D_GROUP, "KV tile should have the same tensor size.");
            for (int bi = 0; bi < blocks_per_tma_load; ++bi)
            {
                int const bounded_block_idx = min(num_valid_kv_blocks - 1, paged_kv_block_idx + bi);

                int32_t const k_paged_block_offset = paged_block_offsets[bounded_block_idx];
                int32_t const v_paged_block_offset = paged_block_offsets[max_blocks_per_sequence + bounded_block_idx];

#pragma unroll
                for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
                {
                    int32_t const k_coords[4]
                        = {di * Kernel_traits::D_PER_GROUP, kv_offset_in_block, bidh, k_paged_block_offset};

                    fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                        __cvta_generic_to_shared(&shared->smem_k[k_barrier_id * TILE_SIZE_K
                            + di * TILE_SIZE_K_PER_D_GROUP + bi * tile_size_k_per_block]),
                        __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
                }

#pragma unroll
                for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
                {
                    int32_t const v_coords[4]
                        = {di * Kernel_traits::D_PER_GROUP, kv_offset_in_block, bidh, v_paged_block_offset};

                    fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_kv,
                        __cvta_generic_to_shared(&shared->smem_v_scratch[v_scratch_barrier_id * TILE_SIZE_V
                            + di * TILE_SIZE_V_PER_D_GROUP + bi * tile_size_k_per_block]),
                        __cvta_generic_to_shared(cbw_v_scratch.barrier_ptr(v_scratch_barrier_id)), v_coords,
                        elect_one_);
                }
            }

            // Do we really need this as we only have one buffer ?
            return v_scratch_barrier_id;
        }

        // Load k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch, typename BufferReaderScratch>
        inline __device__ int load_kv(int bidh, int h, int h_kv, int kv_step_idx, cudaTmaDesc const* desc_kv,
            cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v,
            BufferWriterScratch& cbw_v_scratch, BufferReaderScratch& cbr_v_scratch)
        {

            if constexpr (DMA_GROUP_TRANSPOSE_V)
            {
                int v_scratch_barrier_id = load_kv_transpose_v_impl(
                    bidh, h, h_kv, kv_step_idx, desc_kv, desc_v, shared, cbw_k, cbw_v, cbw_v_scratch, cbr_v_scratch);
                return v_scratch_barrier_id;
            }
            else
            {
                load_kv_impl(bidh, h, h_kv, kv_step_idx, desc_kv, desc_v, shared, cbw_k, cbw_v);
                return 0;
            }
        }

        // Load contiguous kv tiles [B, S, 2, H, D] from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch, typename BufferReaderScratch>
        inline __device__ int load_contiguous_kv(int bidh, int h, int h_kv, int kv_step_idx, cudaTmaDesc const* desc_kv,
            Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v, BufferWriterScratch& cbw_v_scratch,
            BufferReaderScratch& cbr_v_scratch)
        {

            if constexpr (DMA_GROUP_TRANSPOSE_V)
            {
                int v_scratch_barrier_id = load_contiguous_kv_transpose_v_impl(
                    bidh, h, h_kv, kv_step_idx, desc_kv, shared, cbw_k, cbw_v, cbw_v_scratch, cbr_v_scratch);
                return v_scratch_barrier_id;
            }
            else
            {
                load_contiguous_kv_impl(bidh, h, h_kv, kv_step_idx, desc_kv, shared, cbw_k, cbw_v);
                return 0;
            }
        }

        // Load paged k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch, typename BufferReaderScratch>
        inline __device__ int load_paged_kv(int bidh, int kv_tile_start_offset, int num_valid_kv_blocks,
            int tokens_per_block_log2, int blocks_per_tma_load, int blocks_per_tma_load_log2,
            int max_blocks_per_sequence, int32_t const* paged_block_offsets, cudaTmaDesc const* desc_kv,
            cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v,
            BufferWriterScratch& cbw_v_scratch, BufferReaderScratch& cbr_v_scratch)
        {

            if constexpr (DMA_GROUP_TRANSPOSE_V)
            {
                int v_scratch_barrier_id
                    = load_paged_kv_transpose_v_impl(bidh, kv_tile_start_offset, num_valid_kv_blocks,
                        tokens_per_block_log2, blocks_per_tma_load, blocks_per_tma_load_log2, max_blocks_per_sequence,
                        paged_block_offsets, desc_kv, shared, cbw_k, cbw_v, cbw_v_scratch, cbr_v_scratch);
                return v_scratch_barrier_id;
            }
            else
            {
                load_paged_kv_impl(bidh, kv_tile_start_offset, num_valid_kv_blocks, tokens_per_block_log2,
                    blocks_per_tma_load, blocks_per_tma_load_log2, max_blocks_per_sequence, paged_block_offsets,
                    desc_kv, desc_v, shared, cbw_k, cbw_v);
                return 0;
            }
        }

        inline __device__ void get_next_tile_id(
            int local_wid, int tiw, uint32_t smem_tile_id, uint32_t* tile_id_counter_ptr)
        {
            if constexpr (DMA_GROUP_TRANSPOSE_V)
            {
                if (elect_one_)
                {
                    tile_id_ = atomicAdd(tile_id_counter_ptr, 1);
                    sts(smem_tile_id, tile_id_);
                }
                fence_view_async_shared();
                named_barrier_wait(SYNC_BARRIER, 128);
                if (tiw == 0)
                {
                    lds(tile_id_, smem_tile_id);
                }
                tile_id_ = __shfl_sync(0xffffffff, tile_id_, 0);
                // only one warp involved when the dma group doesn't need to transpose the v tile.
            }
            else
            {
                if (elect_one_)
                {
                    tile_id_ = atomicAdd(tile_id_counter_ptr, 1);
                }
                tile_id_ = __shfl_sync(0xffffffff, tile_id_, 0);
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////

    struct Host
    {
        Host() {}

        // Set TMA descriptors on host, and launch as __grid_constant__.
        // Paged KV FMHA parameters.
        void init_params(bert::Fused_multihead_attention_params_v2& params,
            bert::Fused_multihead_attention_launch_params const& launch_params, cudaStream_t stream) const
        {
            if (launch_params.attention_input_layout == fmha::Attention_input_layout::PACKED_QKV)
            {
                // Packed qkv tma descriptors (continuous buffer).
                fmha::Multiple_tma_descriptor<4> qkv_tma_descriptor;

                // Per batch tensor size.
                uint32_t tensor_size_qkv[4];
                // Stride size in bytes. Assumes least significant dim is 1 (?)
                uint64_t tensor_size_qk[3], tensor_size_v[3];
                uint32_t v_offset;
                // Total sequence length.
                int const total_seqlen = params.is_s_padded ? (params.b * params.s) : launch_params.total_q_seqlen;
                tensor_size_qkv[0] = params.d; // params.d;
                tensor_size_qkv[3] = total_seqlen;
                tensor_size_qk[0] = params.d * Kernel_traits::ELEMENT_BYTES;
                tensor_size_qk[2] = params.qkv_stride_in_bytes;
                tensor_size_v[1] = 0;
                tensor_size_v[2] = params.qkv_stride_in_bytes;
                if (params.h_kv < params.h)
                {
                    // Take MQA as non-heads-interleaved.
                    tensor_size_qkv[1] = params.h + params.h_kv;
                    tensor_size_qkv[2] = 1;
                    tensor_size_qk[1] = 0;
                    tensor_size_v[0] = params.dv * Kernel_traits::ELEMENT_BYTES;
                    v_offset = (params.h + params.h_kv) * params.d * Kernel_traits::ELEMENT_BYTES;
                }
                else if (HEADS_INTERLEAVED)
                {
                    tensor_size_qkv[1] = 2;
                    tensor_size_qkv[2] = params.h;
                    tensor_size_qk[1] = (2 * params.d + params.dv) * Kernel_traits::ELEMENT_BYTES;
                    tensor_size_v[0] = tensor_size_qk[1];
                    v_offset = 2 * params.d * Kernel_traits::ELEMENT_BYTES;
                }
                else
                {
                    tensor_size_qkv[1] = params.h;
                    tensor_size_qkv[2] = 2;
                    tensor_size_qk[1] = params.h * tensor_size_qk[0];
                    tensor_size_v[0] = params.dv * Kernel_traits::ELEMENT_BYTES;
                    v_offset = 2 * params.h * params.d * Kernel_traits::ELEMENT_BYTES;
                }

                // O : [TOTAL, 1, h, d]
                uint32_t tensor_size_o[4];
                tensor_size_o[0] = params.dv;
                tensor_size_o[1] = params.h;
                tensor_size_o[2] = 1;
                tensor_size_o[3] = total_seqlen;

                // Box size for k and v.
                uint32_t box_size[4];
                // Update this on device?
                box_size[2] = 1;
                box_size[1] = 1;
                box_size[0] = Kernel_traits::D_PER_GROUP;

                uint64_t tensor_stride_o[3];
                tensor_stride_o[0] = tensor_size_o[0] * Kernel_traits::ELEMENT_BYTES; // dv
                tensor_stride_o[1] = tensor_size_o[1] * tensor_stride_o[0];           // dv*h
                tensor_stride_o[2] = tensor_size_o[2] * tensor_stride_o[1];           // dv*h*1

                // Traversal stride.
                uint32_t traversal_stride_qkv[4] = {1, 1, 1, 1};
                uint32_t traversal_stride_o[4] = {1, 1, 1, 1};

                // OOB fill zeros.
                uint32_t oob_fill = 0;

                // FP32 to TF32 conversion disabled.
                uint32_t fp32_to_tf32 = 0;

                // GMMA descriptor mode.
                static constexpr int D_BYTES_PER_GROUP = Kernel_traits::D_BYTES_PER_GROUP;
                static constexpr fmha::cudaTmaDescSwizzle swizzle_mode
                    = (D_BYTES_PER_GROUP > 64        ? fmha::cudaTmaDescSwizzle::SWIZZLE_128B
                            : D_BYTES_PER_GROUP > 32 ? fmha::cudaTmaDescSwizzle::SWIZZLE_64B
                                                     : fmha::cudaTmaDescSwizzle::SWIZZLE_32B);

                static_assert(STEP_KV <= 256 && STEP_Q <= 256, "max box size is 256");

                // QKV [TOTAL, 3, h, d].
                tensor_size_qkv[3] = params.is_s_padded ? (params.b * params.s) : launch_params.total_q_seqlen;
                tensor_size_o[3] = tensor_size_qkv[3];

                // QKV ptr.
                char* qkv_ptr = reinterpret_cast<char*>(params.qkv_ptr);
                char* o_ptr = reinterpret_cast<char*>(params.o_ptr);

                // Desc Format (data type).
                static constexpr fmha::cudaTmaDescFormat desc_format = (Kernel_traits::ELEMENT_BYTES == 1)
                    ? fmha::cudaTmaDescFormat::U8
                    : fmha::cudaTmaDescFormat::F16_RN;

                // Q: STEP_Q.
                box_size[3] = STEP_Q;
                qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_size_qk,
                    traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &params.tma_desc_q);

                // O: 16
                box_size[3] = 16;
                if (Kernel_traits::USE_TMA_STORE)
                {
                    qkv_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format,
                        fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                        fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_o, tensor_stride_o,
                        traversal_stride_o, box_size, oob_fill, fp32_to_tf32, &params.tma_desc_o);
                }

                // K: STEP_KV.
                box_size[3] = STEP_KV;
                qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_size_qk,
                    traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &params.tma_desc_kv);

                // V: STEP_KV.
                tensor_size_qkv[0] = params.dv;
                tensor_size_qkv[1] = params.h_kv;
                tensor_size_qkv[2] = 1;

                qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr + v_offset, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_size_v,
                    traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &params.tma_desc_v);
            }
            else
            {
                // Separate contiguous q, contiguous kv, and paged kv tma descriptors.
                fmha::Multiple_tma_descriptor<4> qo_tma_descriptor;
                fmha::Multiple_tma_descriptor<4> contiguous_kv_tma_descriptor;
                fmha::Multiple_tma_descriptor<4> paged_kv_tma_descriptor;
                // params.b * 2 * params.paged_kv_cache.mMaxBlocksPerSeq
                //  Per batch tensor size.
                uint32_t tensor_size_qo[4];
                tensor_size_qo[3] = params.is_s_padded ? params.b * params.s : launch_params.total_q_seqlen;
                tensor_size_qo[2] = 1;
                tensor_size_qo[1] = params.h;
                tensor_size_qo[0] = params.d; // params.d;

                // Box size for q and o.
                uint32_t box_size_qo[4];
                box_size_qo[3] = STEP_Q;
                box_size_qo[2] = 1;
                box_size_qo[1] = 1;
                box_size_qo[0] = Kernel_traits::D_PER_GROUP;

                // Stride size in bytes. Assumes least significant dim is 1 (?)
                uint64_t tensor_stride_qo[3];
                tensor_stride_qo[0] = tensor_size_qo[0] * Kernel_traits::ELEMENT_BYTES; // d
                tensor_stride_qo[1] = tensor_size_qo[1] * tensor_stride_qo[0];          // d*h
                tensor_stride_qo[2] = tensor_size_qo[2] * tensor_stride_qo[1];          // d*h*3

                // Traversal stride.
                uint32_t traversal_stride[4] = {1, 1, 1, 1};

                // OOB fill zeros.
                uint32_t oob_fill = 0;

                // FP32 to TF32 conversion disabled.
                uint32_t fp32_to_tf32 = 0;

                // GMMA descriptor mode.
                static constexpr int D_BYTES_PER_GROUP = Kernel_traits::D_BYTES_PER_GROUP;
                static constexpr fmha::cudaTmaDescSwizzle swizzle_mode
                    = (D_BYTES_PER_GROUP > 64        ? fmha::cudaTmaDescSwizzle::SWIZZLE_128B
                            : D_BYTES_PER_GROUP > 32 ? fmha::cudaTmaDescSwizzle::SWIZZLE_64B
                                                     : fmha::cudaTmaDescSwizzle::SWIZZLE_32B);

                static_assert(STEP_KV <= 256 && STEP_Q <= 256, "max box size is 256");

                // Q ptr.
                char* q_ptr = reinterpret_cast<char*>(params.q_ptr);

                // Desc Format (data type).
                static constexpr fmha::cudaTmaDescFormat desc_format = (Kernel_traits::ELEMENT_BYTES == 1)
                    ? fmha::cudaTmaDescFormat::U8
                    : fmha::cudaTmaDescFormat::F16_RN;

                // Q: STEP_Q.
                qo_tma_descriptor.set_tma_desctriptor(q_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo, traversal_stride,
                    box_size_qo, oob_fill, fp32_to_tf32, &params.tma_desc_q);

                // O ptr.
                char* o_ptr = reinterpret_cast<char*>(params.o_ptr);

                // O: 16
                box_size_qo[3] = 16;
                if (Kernel_traits::USE_TMA_STORE)
                {
                    qo_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format,
                        fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                        fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo,
                        traversal_stride, box_size_qo, oob_fill, fp32_to_tf32, &params.tma_desc_o);
                }

                // Contiguous KV: [B, S, 2, H, D].
                if (launch_params.attention_input_layout == fmha::Attention_input_layout::CONTIGUOUS_Q_KV)
                {

                    // Total sequence length.
                    int const total_seqlen = params.is_s_padded ? (params.b * params.s) : launch_params.total_kv_seqlen;
                    uint32_t tensor_size_kv[4];
                    tensor_size_kv[3] = total_seqlen;
                    tensor_size_kv[2] = 2;
                    tensor_size_kv[1] = params.h_kv;
                    tensor_size_kv[0] = params.d;

                    // Box size for k and v.
                    uint32_t box_size_kv[4];
                    box_size_kv[3] = int32_t(STEP_KV);
                    box_size_kv[2] = 1;
                    box_size_kv[1] = 1;
                    box_size_kv[0] = Kernel_traits::D_PER_GROUP;

                    // Stride size in bytes. Assumes least significant dim is 1 (?)
                    uint64_t tensor_stride_kv[3];
                    tensor_stride_kv[0] = tensor_size_kv[0] * Kernel_traits::ELEMENT_BYTES; // d
                    tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];          // d*h_kv
                    tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];          // d*h_kv*2

                    // Contiguous KV pool tma descriptors.
                    contiguous_kv_tma_descriptor.set_tma_desctriptor(reinterpret_cast<char*>(params.kv_ptr),
                        desc_format, fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                        fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_kv, tensor_stride_kv,
                        traversal_stride, box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_kv);
                }
                else
                {
                    // Paged KV: [UINT32_MAX, H, TokensPerBlock, D]
                    // Per batch tensor size.
                    uint32_t tensor_size_kv[4];
                    // The original code is:
                    // tensor_size_kv[3] = params.b * 2 * params.paged_kv_cache.mMaxBlocksPerSeq;
                    // If d != dv and v is not padded, then the code should be:
                    // tensor_size_kv[3] = params.b * params.paged_kv_cache.mMaxBlocksPerSeq
                    //     * ((params.d + params.dv) / std::gcd(params.d, params.dv));
                    // TensorRT-LLM uses:
                    // tensor_size_kv[3] = mLaunchParams.total_device_memory /
                    // mKernelParams.paged_kv_cache.mBytesPerBlock;
                    // I think the simplest way is:
                    tensor_size_kv[3] = INT_MAX;
                    tensor_size_kv[2] = params.h_kv;
                    tensor_size_kv[1] = params.paged_kv_cache.mTokensPerBlock;
                    tensor_size_kv[0] = params.d; // params.d;

                    // Box size for k and v.
                    uint32_t box_size_kv[4];
                    box_size_kv[3] = 1;
                    box_size_kv[2] = 1;
                    box_size_kv[1] = std::min(params.paged_kv_cache.mTokensPerBlock, int32_t(STEP_KV));
                    box_size_kv[0] = Kernel_traits::D_PER_GROUP;

                    assert(int32_t(STEP_KV) % params.paged_kv_cache.mTokensPerBlock == 0
                        || params.paged_kv_cache.mTokensPerBlock % int32_t(STEP_KV) == 0);
                    params.blocks_per_tma_load = std::max(1, int32_t(STEP_KV) / params.paged_kv_cache.mTokensPerBlock);
                    params.blocks_per_tma_load_log2 = log2(params.blocks_per_tma_load);

                    // Stride size in bytes. Assumes least significant dim is 1 (?)
                    uint64_t tensor_stride_kv[3];
                    tensor_stride_kv[0] = tensor_size_kv[0] * Kernel_traits::ELEMENT_BYTES; // d
                    // The original code is:
                    // tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];   // d*mTokensPerBlock
                    // tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];   // d*mTokensPerBlock*h
                    // This can be simplified to:
                    tensor_stride_kv[1] = params.kv_stride_in_bytes;
                    tensor_stride_kv[2] = params.paged_kv_cache.mBytesPerBlock;

                    // Paged KV pool tma descriptors.
                    paged_kv_tma_descriptor.set_tma_desctriptor(reinterpret_cast<char*>(params.paged_kv_cache.mPoolPtr),
                        desc_format, fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                        fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_kv, tensor_stride_kv,
                        traversal_stride, box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_kv);
#ifndef GENERATE_CUBIN
                    tensor_size_kv[0] = params.dv;
                    tensor_stride_kv[0] = tensor_size_kv[0] * Kernel_traits::ELEMENT_BYTES; // dv
                    tensor_stride_kv[1] = params.v_stride_in_bytes;                         // dv*mTokensPerBlock

                    paged_kv_tma_descriptor.set_tma_desctriptor(reinterpret_cast<char*>(params.paged_kv_cache.mPoolPtr),
                        desc_format, fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                        fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_kv, tensor_stride_kv,
                        traversal_stride, box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_v);
#endif
                }
            }
        }
    };
};

} // namespace ws
} // namespace fmha
