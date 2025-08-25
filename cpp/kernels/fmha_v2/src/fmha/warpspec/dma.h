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
                    : max(0, q_step_offset + 1 - params.sliding_window_size);
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
                cudaTmaDesc const* desc_k = &params.tma_desc_k;
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
                sum_s_kv_ = sum_s_q_;

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
                    load_q(bidh, (q_step_idx + 0 + q_step_offset) * STEP_Q, desc_q, shared->smem_q[0], cbw0);
                    load_q(bidh, (q_step_idx + 1 + q_step_offset) * STEP_Q, desc_q, shared->smem_q[1], cbw1);

                    // Q step bound is 2 tiles away at this moment because of 2x1 math warpgroup
                    int const q_step_end = (q_step_idx + q_step_offset + 2) * STEP_Q - 1;

                    // The kv tile idx range for this q step.
                    auto const [kv_idx_start, kv_idx_end]
                        = compute_kv_tile_idx(params, (q_step_idx + q_step_offset) * STEP_Q, q_step_end, kv_steps);

                    // Iterate over the kv tiles for this q step.
                    for (int kv_step_idx = kv_idx_start; kv_step_idx < kv_idx_end; kv_step_idx++)
                    {
                        int bar_id = load_kv(bidh / params.h_q_per_kv, kv_step_idx * STEP_KV, desc_k, desc_v, shared,
                            cbw_k, cbw_v, cbw_v_scratch);

                        // Opportunistically hide headinfo in the shadow of UTMALDGs of the QKV tensor
                        if (q_step_idx == 0 && kv_step_idx == kv_idx_start)
                        {
                            // Send head info.
                            typename Shared::Head_info info{q_steps,
                                // q, and kv have the same length.
                                q_tile_offset, USE_CUSTOM_MASK ? sum_mask_s : q_tile_offset, kv_steps,
                                // q, and kv have the same length.
                                actual_seqlen, actual_seqlen, sum_s_q_ * params.h + bidh, bidh, bidb};
                            // NOTE(tizheng): The need for the sync after consumer bar wait is to avoid a deadlock
                            // hazard when DMA thread 0 is ahead of other DMA threads. For example: DMA thread 0 have
                            // finished consumer bar wait phase 0 and producer bar arrive phase 0, and then MMA warps
                            // have finished producer bar wait phase 0 and consumer bar arrive phase 1. At this time
                            // other DMA threads start consumer bar wait phase 0. It will never become ready. DMA warps
                            // then fail to continue to the next loop.
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
                cudaTmaDesc const* desc_k = &params.tma_desc_k;
                cudaTmaDesc const* desc_v = &params.tma_desc_v;

                int32_t const* paged_block_offsets
                    = params.paged_kv_cache.mBlockOffsets + bidb * 2 * params.paged_kv_cache.mMaxBlocksPerSeq;

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

                for (int q_step_idx = 0; q_step_idx < q_steps && actual_kv_seqlen > 0; q_step_idx += 2)
                {
                    load_q(bidh, q_step_idx * STEP_Q + local_q_tile_offset, desc_q, shared->smem_q[0], cbw0);
                    load_q(bidh, (q_step_idx + 1) * STEP_Q + local_q_tile_offset, desc_q, shared->smem_q[1], cbw1);

                    // Q step end is 2 tiles away at this moment because of 2x1 math warpgroup
                    int const q_step_end = (q_step_idx + 2) * STEP_Q - 1 + q_tile_offset;

                    // The kv tile idx range for this q step.
                    auto const [kv_idx_start, kv_idx_end]
                        = compute_kv_tile_idx(params, q_step_idx * STEP_Q + q_tile_offset, q_step_end, kv_steps);

                    // Iterate over the kv tiles for this q step.
                    for (int kv_step_idx = kv_idx_start; kv_step_idx < kv_idx_end; kv_step_idx++)
                    {
                        // The barrier id.
                        int bar_id;
                        // Load paged kv input.
                        if constexpr (PAGED_KV_INPUT)
                        {
                            bar_id = load_paged_kv(bidh_kv, kv_step_idx * STEP_KV, num_valid_kv_blocks,
                                params.paged_kv_cache.mTokensPerBlockLog2, params.blocks_per_tma_load,
                                params.blocks_per_tma_load_log2, params.paged_kv_cache.mMaxBlocksPerSeq,
                                paged_block_offsets, desc_k, desc_v, shared, cbw_k, cbw_v, cbw_v_scratch);
                        }
                        else
                        {
                            bar_id = load_kv(
                                bidh_kv, kv_step_idx * STEP_KV, desc_k, desc_v, shared, cbw_k, cbw_v, cbw_v_scratch);
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
            int bidh, int q_tile_start_offset, cudaTmaDesc const* desc_q, Smem_q& smem_q, BufferWriter& cbw)
        {

            int barrier_id = cbw.tmaReserve(elect_one_, TILE_SIZE_Q * Kernel_traits::ELEMENT_BYTES);

            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                const int32_t coords[3] = {di * Kernel_traits::D_PER_GROUP, bidh, sum_s_q_ + q_tile_start_offset};
                fmha::utmaldg<3, fmha::cudaTmaDescType::TILED, false>(desc_q,
                    __cvta_generic_to_shared(&smem_q[barrier_id * TILE_SIZE_Q + di * TILE_SIZE_Q_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw.barrier_ptr(barrier_id)), coords, elect_one_);
            }
        }

#define PREPARE_KV_BUFFER()                                                                                            \
    int k_barrier_id = cbw_k.tmaReserve(elect_one_, (TILE_SIZE_K) *Kernel_traits::ELEMENT_BYTES);                      \
                                                                                                                       \
    int v_barrier_id;                                                                                                  \
    void* v_barrier_ptr;                                                                                               \
    typename Kernel_traits::Element_data_type* v_smem;                                                                 \
                                                                                                                       \
    if constexpr (DMA_GROUP_TRANSPOSE_V)                                                                               \
    {                                                                                                                  \
        v_barrier_id = cbw_v_scratch.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);              \
        v_barrier_ptr = cbw_v_scratch.barrier_ptr(v_barrier_id);                                                       \
        v_smem = shared->smem_v_scratch.data();                                                                        \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        v_barrier_id = cbw_v.tmaReserve(elect_one_, (TILE_SIZE_V) *Kernel_traits::ELEMENT_BYTES);                      \
        v_barrier_ptr = cbw_v.barrier_ptr(v_barrier_id);                                                               \
        v_smem = shared->smem_v.data();                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP);

        // Load k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch>
        inline __device__ int load_kv(int bidh_kv, int kv_tile_start_offset, cudaTmaDesc const* desc_k,
            cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v,
            BufferWriterScratch& cbw_v_scratch)
        {
            PREPARE_KV_BUFFER()

            // split D into multiple groups in order to satisfy the TMA 128B sizzle mode
#pragma unroll
            for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
            {
                const int32_t k_coords[3]
                    = {di * Kernel_traits::D_PER_GROUP, bidh_kv, sum_s_kv_ + kv_tile_start_offset};

                fmha::utmaldg<3, fmha::cudaTmaDescType::TILED, false>(desc_k,
                    __cvta_generic_to_shared(
                        &shared->smem_k[k_barrier_id * TILE_SIZE_K + di * TILE_SIZE_K_PER_D_GROUP]),
                    __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
            }

#pragma unroll
            for (int di = 0; di < Kernel_traits::DV_GROUPS; ++di)
            {
                const int32_t v_coords[3]
                    = {di * Kernel_traits::D_PER_GROUP, bidh_kv, sum_s_kv_ + kv_tile_start_offset};

                fmha::utmaldg<3, fmha::cudaTmaDescType::TILED, false>(desc_v,
                    __cvta_generic_to_shared(&v_smem[v_barrier_id * TILE_SIZE_V + di * TILE_SIZE_V_PER_D_GROUP]),
                    __cvta_generic_to_shared(v_barrier_ptr), v_coords, elect_one_);
            }

            return v_barrier_id;
        }

        // Load paged k,v tiles from gmem to smem by TMA.
        template <typename BufferWriter, typename BufferWriterScratch>
        inline __device__ int load_paged_kv(int bidh_kv, int kv_tile_start_offset, int num_valid_kv_blocks,
            int tokens_per_block_log2, int blocks_per_tma_load, int blocks_per_tma_load_log2,
            int max_blocks_per_sequence, int32_t const* paged_block_offsets, cudaTmaDesc const* desc_k,
            cudaTmaDesc const* desc_v, Shared* shared, BufferWriter& cbw_k, BufferWriter& cbw_v,
            BufferWriterScratch& cbw_v_scratch)
        {
            PREPARE_KV_BUFFER()

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

                const int32_t k_paged_block_offset = paged_block_offsets[bounded_block_idx];
                const int32_t v_paged_block_offset = paged_block_offsets[max_blocks_per_sequence + bounded_block_idx];

#pragma unroll
                for (int di = 0; di < Kernel_traits::D_GROUPS; ++di)
                {
                    const int32_t k_coords[4]
                        = {di * Kernel_traits::D_PER_GROUP, kv_offset_in_block, bidh_kv, k_paged_block_offset};

                    fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_k,
                        __cvta_generic_to_shared(&shared->smem_k[k_barrier_id * TILE_SIZE_K
                            + di * TILE_SIZE_K_PER_D_GROUP + bi * tile_size_k_per_block]),
                        __cvta_generic_to_shared(cbw_k.barrier_ptr(k_barrier_id)), k_coords, elect_one_);
                }

#pragma unroll
                for (int di = 0; di < Kernel_traits::DV_GROUPS; ++di)
                {
                    const int32_t v_coords[4]
                        = {di * Kernel_traits::D_PER_GROUP, kv_offset_in_block, bidh_kv, v_paged_block_offset};

                    fmha::utmaldg<4, fmha::cudaTmaDescType::TILED, false>(desc_v,
                        __cvta_generic_to_shared(&v_smem[v_barrier_id * TILE_SIZE_V + di * TILE_SIZE_V_PER_D_GROUP
                            + bi * tile_size_k_per_block]),
                        __cvta_generic_to_shared(v_barrier_ptr), v_coords, elect_one_);
                }
            }

            return v_barrier_id;
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
                for (int dgroup_idx = 0; dgroup_idx < Kernel_traits::DV_GROUPS; dgroup_idx++)
                {
                    // Src smem block is k first then d
                    uint32_t src_offset = (kgroup_idx * Kernel_traits::BMM2_K_PER_GROUP * Kernel_traits::D_PER_GROUP
                                              + dgroup_idx * Kernel_traits::D_PER_GROUP * Kernel_traits::STEP_KV)
                        * Kernel_traits::ELEMENT_BYTES;

                    // Dst smem block is d first then k
                    uint32_t dst_offset = (dgroup_idx * Kernel_traits::BMM2_K_PER_GROUP * Kernel_traits::D_PER_GROUP
                                              + kgroup_idx * Kernel_traits::BMM2_K_PER_GROUP * Kernel_traits::DV)
                        * Kernel_traits::ELEMENT_BYTES;

                    transposer.template transpose_<false>(smem_v_src + src_offset, smem_v_dst + dst_offset);
                }
            }

            fence_view_async_shared();                                  // Commit STSM
            named_barrier_wait(SYNC_BARRIER, NUM_THREADS_IN_DMA_GROUP); // Sync before signaling
            cbw_v.threadCommit(elect_one_, v_barrier_id);               // Signal readiness
            cbr_v_scratch.pop(elect_one_);                              // Advance to next phase
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
            const uint32_t d = params.d;
            const uint32_t dv = params.dv;
            const uint32_t h = params.h;
            const uint32_t h_kv = params.h_kv;

            // Total sequence length.
            const uint32_t total_seqlen = params.is_s_padded ? (params.b * params.s) : launch_params.total_q_seqlen;

            // O Layout: [total_seqlen, H, DV]
            // Per batch tensor size.
            uint32_t tensor_size_o[3] = {dv, h, total_seqlen};

            // Stride size in bytes. Assumes least significant dim is 1
            uint64_t tensor_stride_o[2] = {dv * Kernel_traits::ELEMENT_BYTES, uint64_t(params.o_stride_in_bytes)};

            // Starting memory address
            char* o_ptr = reinterpret_cast<char*>(params.o_ptr);

            // Box size of TMA
            uint32_t box_size_o[3] = {Kernel_traits::D_PER_GROUP, 1, 16};

            // Traversal stride.
            uint32_t traversal_stride[3] = {1, 1, 1};

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

            // Desc Format (data type).
            static constexpr fmha::cudaTmaDescFormat desc_format
                = (Kernel_traits::ELEMENT_BYTES == 1) ? fmha::cudaTmaDescFormat::U8 : fmha::cudaTmaDescFormat::F16_RN;

            fmha::Multiple_tma_descriptor<3> qo_tma_descriptor;

            // TMA O
            if (Kernel_traits::USE_TMA_STORE)
            {
                qo_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_o, tensor_stride_o, traversal_stride,
                    box_size_o, oob_fill, fp32_to_tf32, &params.tma_desc_o);
            }

            auto const layout = launch_params.attention_input_layout;

            // Q always uses 3D tensor
            uint32_t tensor_size_q[3] = {d, h, total_seqlen};

            uint64_t tensor_stride_q[2] = {d * Kernel_traits::ELEMENT_BYTES, uint64_t(params.q_stride_in_bytes)};

            char* q_ptr = reinterpret_cast<char*>(
                layout == fmha::Attention_input_layout::PACKED_QKV ? params.qkv_ptr : params.q_ptr);

            uint32_t box_size_q[3] = {Kernel_traits::D_PER_GROUP, 1, STEP_Q};

            if (layout == fmha::Attention_input_layout::Q_PAGED_KV)
            {
                // KV in q_paged_kv uses 4D tensor
                // Layout: [INT32_MAX, H_KV, TokensPerBlock, D]
                const uint32_t tokens_per_block = params.paged_kv_cache.mTokensPerBlock;
                uint32_t tensor_size_k[4] = {d, tokens_per_block, h_kv, INT_MAX};
                uint32_t tensor_size_v[4] = {dv, tokens_per_block, h_kv, INT_MAX};

                uint64_t tensor_stride_k[3];
                tensor_stride_k[0] = params.k_stride_in_bytes / tokens_per_block; // d
                tensor_stride_k[1] = params.k_stride_in_bytes;                    // d * 64
                tensor_stride_k[2] = params.paged_kv_cache.mBytesPerBlock;
                uint64_t tensor_stride_v[3];
                // we cannot use dv * Kernel_traits::ELEMENT_BYTES because V may be padded (MLA)
                tensor_stride_v[0] = params.v_stride_in_bytes / tokens_per_block; // dv
                tensor_stride_v[1] = params.v_stride_in_bytes;                    // dv * 64
                tensor_stride_v[2] = params.paged_kv_cache.mBytesPerBlock;

                char* kv_ptr = reinterpret_cast<char*>(params.paged_kv_cache.mPoolPtr);

                uint32_t box_size_kv[4]
                    = {Kernel_traits::D_PER_GROUP, std::min<uint32_t>(tokens_per_block, STEP_KV), 1, 1};

                assert(STEP_KV % tokens_per_block == 0 || tokens_per_block % STEP_KV == 0);
                params.blocks_per_tma_load = std::max<uint32_t>(1, STEP_KV / tokens_per_block);
                params.blocks_per_tma_load_log2 = log2(params.blocks_per_tma_load);

                uint32_t traversal_stride[4] = {1, 1, 1, 1};

                fmha::Multiple_tma_descriptor<4> kv_tma_descriptor;
                // K
                kv_tma_descriptor.set_tma_desctriptor(kv_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_k, tensor_stride_k, traversal_stride,
                    box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_k);
                // V
                kv_tma_descriptor.set_tma_desctriptor(kv_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_v, tensor_stride_v, traversal_stride,
                    box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_v);
            }
            else
            {
                // Otherwise KV uses 3D tensor
                uint32_t tensor_size_k[3] = {d, h_kv, total_seqlen};
                uint32_t tensor_size_v[3] = {dv, h_kv, total_seqlen};

                uint64_t tensor_stride_k[2] = {d * Kernel_traits::ELEMENT_BYTES, uint64_t(params.k_stride_in_bytes)};
                uint64_t tensor_stride_v[2] = {dv * Kernel_traits::ELEMENT_BYTES, uint64_t(params.v_stride_in_bytes)};

                uint32_t box_size_kv[3] = {Kernel_traits::D_PER_GROUP, 1, STEP_KV};

                char *k_ptr, *v_ptr;

                if (layout == fmha::Attention_input_layout::PACKED_QKV)
                {
                    if (!HEADS_INTERLEAVED || h != h_kv)
                    {
                        // Layout: [total_seqlen, (H, D) + (H_KV, D) + (H_KV, DV)]
                        // All of MHA in TRTLLM is in this layout,
                        // and MQA/GQA must use this layout.
                        k_ptr = q_ptr + h * d * Kernel_traits::ELEMENT_BYTES;
                        v_ptr = k_ptr + h_kv * d * Kernel_traits::ELEMENT_BYTES;
                    }
                    else
                    {
                        // Layout: [total_seqlen, H, D + D + DV]
                        // Currently only used in MHA in fmha_v2 tests.
                        tensor_stride_q[0] = tensor_stride_k[0] = tensor_stride_v[0]
                            = (2 * d + dv) * Kernel_traits::ELEMENT_BYTES;
                        k_ptr = q_ptr + d * Kernel_traits::ELEMENT_BYTES;
                        v_ptr = k_ptr + d * Kernel_traits::ELEMENT_BYTES;
                    }
                }
                else if (layout == fmha::Attention_input_layout::CONTIGUOUS_Q_KV)
                {
                    k_ptr = reinterpret_cast<char*>(params.kv_ptr);
                    v_ptr = k_ptr + h_kv * d * Kernel_traits::ELEMENT_BYTES;
                }
                else if (layout == fmha::Attention_input_layout::SEPARATE_Q_K_V)
                {
                    k_ptr = reinterpret_cast<char*>(params.k_ptr);
                    v_ptr = reinterpret_cast<char*>(params.v_ptr);
                }

                fmha::Multiple_tma_descriptor<3> kv_tma_descriptor;
                // K
                kv_tma_descriptor.set_tma_desctriptor(k_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_k, tensor_stride_k, traversal_stride,
                    box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_k);
                // V
                kv_tma_descriptor.set_tma_desctriptor(v_ptr, desc_format,
                    fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
                    fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_v, tensor_stride_v, traversal_stride,
                    box_size_kv, oob_fill, fp32_to_tf32, &params.tma_desc_v);
            }
            // Q
            qo_tma_descriptor.set_tma_desctriptor(q_ptr, desc_format, fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                swizzle_mode, fmha::cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_q, tensor_stride_q,
                traversal_stride, box_size_q, oob_fill, fp32_to_tf32, &params.tma_desc_q);
        }
    };
};

} // namespace ws
} // namespace fmha
