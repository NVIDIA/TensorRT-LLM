/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "fmha/alibi_params.h"
#include "fmha/hopper/fragment.h"
#include "fmha/hopper/utils_warpgroup.h"
#include "fmha/softmax.h"
#include "fmha/warpspec/circular_buffer.h"
#include "fmha/warpspec/dma.h"
#include "fmha/warpspec/epilogue.h"

namespace fmha
{
namespace ws
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // Template instruction traits to specialize structs
    template <int, int, int, bool, bool> class Instruction_traits,
    // Kernel Traits
    typename Kernel_traits>
struct Compute
{

    // The shared struct.
    using Shared = typename Kernel_traits::Shared;

    // The q, or kv tile reader.
    using Circular_buffer_q_reader = typename Kernel_traits::Circular_buffer_q_reader;
    using Circular_buffer_kv_reader = typename Kernel_traits::Circular_buffer_kv_reader;

    // The instruction traits for BMM1.
    using Traits_p = typename Kernel_traits::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Kernel_traits::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The Q shared memory tile.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    // The K shared memory tile.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    // The V shared memory tile.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Kernel_traits::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Kernel_traits::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Kernel_traits::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // The global memory tile for storing BMM2 output.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;

    // Softmax
    using Softmax = Softmax<Instruction_traits, Kernel_traits>;

    // BMM2 epilogue
    using Tile_o_epilogue = Tile_o_epilogue<Instruction_traits, Kernel_traits>;

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

    // The number of compute groups (currently fixed at 2).
    enum
    {
        NUM_COMPUTE_GROUPS = Kernel_traits::NUM_COMPUTE_GROUPS
    };

    // Whether we skip those masked tiles when causal mask is enabled ?
    enum
    {
        SKIP_CAUSAL_MASK_TILES = Kernel_traits::CAUSAL_MASK && !Kernel_traits::USE_CUSTOM_MASK
    };

    // Whether we attend to the specific sliding window or chunk ?
    enum
    {
        SLIDING_OR_CHUNKED_ATTENTION = Kernel_traits::SLIDING_OR_CHUNKED_ATTENTION
    };

    // Whether use the bidirectional sliding window attention or not.
    enum
    {
        BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION = Kernel_traits::BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION
    };

    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    enum
    {
        APPLY_ALIBI = Kernel_traits::APPLY_ALIBI
    };

    // Do we use custom mask input ?
    enum
    {
        USE_CUSTOM_MASK = Kernel_traits::USE_CUSTOM_MASK
    };

    // Do we always need to apply the mask ?
    enum
    {
        ALWAYS_APPLY_MASK = APPLY_ALIBI || USE_CUSTOM_MASK
    };

    // Enable mutex for overlapping mma and softmax instructions.
    enum
    {
        ENABLE_MUTEX = Kernel_traits::ENABLE_MUTEX
    };

    // The head_dimension groups.
    enum
    {
        D_GROUPS = Kernel_traits::D_GROUPS
    };

    // The MMA_K groups (corresponding to head_dimension groups).
    enum
    {
        BMM1_MMAS_K_GROUPS = Kernel_traits::D_GROUPS
    };

    // The number of MMAS_K for each head_dimension group.
    enum
    {
        BMM1_MMAS_K_PER_GROUP = Mma_tile_p::MMAS_K / BMM1_MMAS_K_GROUPS
    };

    // The MMA_K groups (corresponding to kv_step groups).
    enum
    {
        BMM2_MMAS_K_GROUPS = Kernel_traits::BMM2_K_GROUPS
    };

    // The number of MMAS_K for each head_dimension group.
    enum
    {
        BMM2_MMAS_K_PER_GROUP = Mma_tile_o::MMAS_K / BMM2_MMAS_K_GROUPS
    };

    // The tile size of V after head_dimension split.
    enum
    {
        TILE_SIZE_V_PER_D_GROUP = STEP_KV * Kernel_traits::D_PER_GROUP
    };

    enum
    {
        TILE_SIZE_V = STEP_KV * Kernel_traits::DV
    };

    enum
    {
        TILE_BYTES_V_PER_D_GROUP = STEP_KV * Kernel_traits::D_BYTES_PER_GROUP
    };

    enum
    {
        TILE_BYTES_V_PER_K_GROUP = BMM2_MMAS_K_PER_GROUP * Kernel_traits::D_BYTES_PER_GROUP
    };

    // Named barrier for inter-warpgroup sync
    enum
    {
        SYNC_BARRIER = Kernel_traits::MMA_SYNC_BARRIER_ID
    };

    // Whether Q and KV is in separate buffer, which means we need to consider different Q and KV lengths.
    enum
    {
        SEPARATE_Q_KV_BUFFER = Kernel_traits::SEPARATE_Q_KV_BUFFER
    };

    enum
    {
        SAGE_BLOCK_SIZE_Q = Kernel_traits::SAGE_BLOCK_SIZE_Q
    };

    enum
    {
        SAGE_BLOCK_SIZE_K = Kernel_traits::SAGE_BLOCK_SIZE_K
    };

    enum
    {
        SAGE_BLOCK_SIZE_V = Kernel_traits::SAGE_BLOCK_SIZE_V
    };

    // SageAttention groups tokens the way the BMM1 fragment distributes them across threads, so a
    // thread never needs more than one scale per axis. Q: the 2 rows a thread owns (quad_row and
    // quad_row + 8, strided). K: one of the SAGE_K_CHUNKS sub-fragments of a thread's key columns.
    enum
    {
        SAGE_Q_TOKENS_PER_SCALE = 2
    };

    enum
    {
        SAGE_K_TOKENS_PER_SCALE = STEP_KV / (4 * Kernel_traits::SAGE_K_CHUNKS)
    };

    enum
    {
        SAGE_Q_SCALES_PER_TILE = STEP_Q / SAGE_Q_TOKENS_PER_SCALE
    };

    enum
    {
        SAGE_K_SCALES_PER_TILE = 4 * Kernel_traits::SAGE_K_CHUNKS
    };

    // The scale buffers are laid out as (H, scales for the whole batch) rather than
    // (B, H, max_nblock): the caller sizes its workspace when only the batch size and the total
    // token count are known, not max_seqlen. A sequence's scales begin at a base derived from its
    // cu_seqlens entry, and each sequence reserves one spare tile so that the scales of a partial
    // final tile -- which the kernel indexes in full -- cannot run into the next sequence.
    enum
    {
        SAGE_Q_SLOTS_PER_SEQ = SAGE_Q_SCALES_PER_TILE
    };

    // K reserves 4 more so its base can be rounded up to a multiple of 4, which keeps the float4
    // scale load aligned. Rounding up costs at most 3, and one spare tile plus 4 covers that.
    enum
    {
        SAGE_K_SLOTS_PER_SEQ = SAGE_K_SCALES_PER_TILE + 4
    };

    static_assert(SAGE_K_SLOTS_PER_SEQ % 4 == 0, "the K scale base must stay 16B-aligned");
    static_assert(SAGE_BLOCK_SIZE_Q <= 0 || SAGE_BLOCK_SIZE_Q == SAGE_Q_TOKENS_PER_SCALE,
        "sage block_q must be 2: Q is quantized per thread (2 strided rows)");
    static_assert(SAGE_BLOCK_SIZE_K <= 0 || SAGE_BLOCK_SIZE_K == SAGE_K_TOKENS_PER_SCALE,
        "sage block_k must be 16: K is quantized per thread sub-fragment");
    static_assert(SAGE_BLOCK_SIZE_V <= 0 || Kernel_traits::SAGE_V_PER_CHANNEL,
        "V is quantized along the channel axis; sage block_v must be 1 or 0");

#define K_TILE_WAIT()                                                                                                  \
    int ready_k = cbr_k.peek();                                                                                        \
    if (!ready_k)                                                                                                      \
    {                                                                                                                  \
        cbr_k.wait();                                                                                                  \
    }

#define KV_TILE_COMPLETE()                                                                                             \
    cbr_k.complete(tidx == 0, cbr_k.ptr());                                                                            \
    cbr_v.complete(tidx == 0, cbr_v.ptr());                                                                            \
    cbr_k.advance();                                                                                                   \
    cbr_v.advance();

#define COMPUTE_SINGLE_TILE(IS_FIRST_COL, APPLY_MASK)                                                                  \
    compute_single_tile<IS_FIRST_COL, APPLY_MASK>(params, ctile_p, softmax, ctile_o, p_max, p_sum, tidx,               \
        actual_kv_seqlen, alibi_head_scale,                                                                            \
        USE_CUSTOM_MASK ? (head_info.mask_sum_s + q_step_idx * STEP_Q + local_q_tile_offset)                           \
                        : (q_step_idx * STEP_Q + head_info.q_tile_offset),                                             \
        kv_step_idx * STEP_KV, sage_scale_base_kv, cbr, cbr_v, mutex_accessor,                                         \
        &shared->skip_softmax_votes[kv_step_idx & 1][warpgroup_id], kv_step_idx == kv_idx_end - 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int div_up(int a, int b)
    {
        return (a + b - 1) / b;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute the kv_left_mask_end and kv_right_mask_start, where mask is applied when kv_idx < kv_left_mask_end or
    // kv_idx >= kv_right_mask_start.
    template <typename Params>
    inline __device__ std::pair<int, int> compute_kv_mask_start_end(
        Params const& params, int const tile_offset_start, int const tile_offset_end, int const kv_idx_end)
    {
        // The kv_left_mask_end is 0 by default.
        int kv_left_mask_end = 0;
        // The kv_right_mask_start is kv_idx_end - 1 by default, which means only the last kv tile is masked.
        int kv_right_mask_start = kv_idx_end - 1;

        // Always apply mask is specified.
        if constexpr (ALWAYS_APPLY_MASK)
        {
            return std::make_pair(0, 0);
        }

        // Is the chunked_attention used ?
        bool is_chunked_attention = params.log2_chunked_attention_size > 0;

        // Handle sliding window or chunked attention masking
        if constexpr (SLIDING_OR_CHUNKED_ATTENTION)
        {
            if constexpr (BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION)
            {
                // Handle bidirectional sliding window attention
                kv_left_mask_end = div_up(tile_offset_end - params.sliding_window_size / 2, STEP_KV);
                kv_right_mask_start
                    = min(kv_idx_end - 1, (tile_offset_start + params.sliding_window_size / 2 + 1) / STEP_KV);
            }
            else if (is_chunked_attention)
            {
                // Handle chunked attention
                kv_left_mask_end = div_up(
                    ((tile_offset_end >> params.log2_chunked_attention_size) << params.log2_chunked_attention_size),
                    STEP_KV);
            }
            else
            {
                kv_left_mask_end = div_up(tile_offset_end + 1 - params.sliding_window_size, STEP_KV);
            }
        }

        // The right mask is needed when causal mask is used.
        if constexpr (SKIP_CAUSAL_MASK_TILES)
        {
            kv_right_mask_start = tile_offset_start / STEP_KV;
        }

        // Return the kv_left_mask_end and kv_right_mask_start.
        return std::make_pair(kv_left_mask_end, kv_right_mask_start);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Params>
    inline __device__ void run(int warpgroup_id, int tidx, Shared* shared, Params const& params)
    {

        auto head_tracker = shared->head_info_tracker[warpgroup_id].createReader();
        auto cbr = shared->tma_q_tracker[warpgroup_id].createReader();

        auto cbr_k = shared->tma_k_tracker.createReader();
        auto cbr_v = shared->tma_v_tracker.createReader();

        // Ctile_p initialize (relies on q_stage, kv_stage).
        char* smem_q = reinterpret_cast<char*>(&shared->smem_q[warpgroup_id][0]);
        char* smem_k = reinterpret_cast<char*>(&shared->smem_k[0]);
        Compute_tile_p ctile_p(smem_q, smem_k);

        // Softmax
        Softmax softmax(params, tidx);

        // Ctile_o initialize (relies on kv_stage).
        uint32_t smem_v = __cvta_generic_to_shared(&shared->smem_v[0]);
        Compute_tile_o ctile_o(0, smem_v);

        // Mutex between two compute groups.
        OrderedMutexAccessor mutex_accessor(shared->compute_mutex, warpgroup_id, SYNC_BARRIER);
        // Notify warpgroup 0 to execute HGMMA first (overlap HGMMA and Softmax Math Instructions).
        if (ENABLE_MUTEX && warpgroup_id == 1 && Kernel_traits::ELEMENT_BYTES == 2)
        {
            mutex_accessor.arrive();
        }

        // While loop for different heads.
        while (true)
        {

            typename Shared::Head_info head_info = head_tracker.pop(true);

            if (head_info.kv_steps == -1)
            {
                break;
            }

            int const kv_steps = head_info.kv_steps;
            int const q_steps = head_info.q_steps;
            int const local_q_tile_offset = head_info.local_q_tile_offset;
            // The global q tile offset (based on past kv cache).
            // Not used by custom mask input.
            int const q_tile_offset = SEPARATE_Q_KV_BUFFER ? head_info.q_tile_offset : head_info.local_q_tile_offset;
            int const actual_q_seqlen = head_info.actual_seqlen;
            // Contiguous QKV FMHA assumes q, and kv have the same sequence length.
            int const actual_kv_seqlen = SEPARATE_Q_KV_BUFFER ? head_info.actual_kv_seqlen : actual_q_seqlen;

            // Update threshold of Skip-Softmax
            if constexpr (Kernel_traits::ENABLE_SKIP_SOFTMAX)
            {
                softmax.skip_softmax_threshold = params.skip_softmax_threshold_scale_factor / actual_kv_seqlen;
            }

            // Calculate the alibi head_scaling_factor.
            float alibi_head_scale
                = APPLY_ALIBI ? get_alibi_head_scaling_factor<AlibiParams>(head_info.bidh, params.alibi_params) : 0.f;
            // pre-compute where this (sequence, head) pair's scales start, for reuse below. Q is
            // quantized per query head, K per key/value head: in MQA/GQA several query heads share
            // one KV head. max_nblock is the per-head stride of the whole batch's scales.
            int sage_scale_base = 0;
            int sage_scale_base_kv = 0;
            if constexpr (Kernel_traits::SAGE_ATTENTION)
            {
                int const bidb = head_info.bidb;
                sage_scale_base = head_info.bidh * params.sage.q.max_nblock
                    + params.cu_q_seqlens[bidb] / SAGE_Q_TOKENS_PER_SCALE + bidb * SAGE_Q_SLOTS_PER_SEQ;
                // Round the dense part up so the base is a multiple of 4; SAGE_K_SLOTS_PER_SEQ is
                // one too, so every sequence's base stays 16B-aligned.
                int const k_dense = params.cu_kv_seqlens[bidb] / SAGE_K_TOKENS_PER_SCALE;
                sage_scale_base_kv = (head_info.bidh / params.h_q_per_kv) * params.sage.k.max_nblock
                    + ((k_dense + 3) & ~3) + bidb * SAGE_K_SLOTS_PER_SEQ;
            }

            // BMM2 epilogue
            Tile_o_epilogue tile_o_epilogue(params, head_info);

            int q_step_idx = warpgroup_id;

            // Compute work.
            for (; q_step_idx < q_steps; q_step_idx += NUM_COMPUTE_GROUPS)
            {

                // Check whether it is a valid run of q steps.
                int const q_offset = q_step_idx * STEP_Q + local_q_tile_offset;
                bool const valid_run = q_offset < actual_q_seqlen;
                // fuse the scale of q into scale_bmm1
                if constexpr (SAGE_BLOCK_SIZE_Q > 0)
                {
                    // I tried another implementation here: store original `scale_bmm1` to a local variable
                    // to avoid frequent `__ldg`. But experiment shows that the current one is faster.
                    // A bit counterintuitive.
                    auto const scale_bmm1 = params.scale_bmm1_d ? __ldg(params.scale_bmm1_d) : params.scale_bmm1;
                    // One scale per thread: it covers exactly the two rows this thread owns. All
                    // four lanes of a quad share quad_row and therefore read the same scale, which
                    // keeps the row-max shuffle reduction consistent. Derived from tidx because
                    // quad_row_ is only initialised for causal/sliding kernels.
                    int const q_pair = (tidx / 32) * 8 + (tidx % 32) / 4;
                    int const idx = sage_scale_base + (q_offset / STEP_Q) * SAGE_Q_SCALES_PER_TILE + q_pair;
                    *(float*) (&softmax.scale_bmm1_)
                        = reinterpret_cast<float const&>(scale_bmm1) * __ldg(&params.sage.q.scales[idx]);
                }

                // KV tile is shared by two q tiles,
                // so we need to consider the last compute group's q tile.
                int const tile_offset_start = q_step_idx * STEP_Q + q_tile_offset;
                int const tile_offset_end = tile_offset_start + STEP_Q - 1;
                int const warpgroup_tile_offset_start = tile_offset_start - warpgroup_id * STEP_Q;
                int const warpgroup_tile_offset_end
                    = tile_offset_start + (NUM_COMPUTE_GROUPS - warpgroup_id) * STEP_Q - 1;

                // Compute the kv_idx start (inclusive) and end (exclusive).
                auto const [kv_idx_start, kv_idx_end] = DMA<Kernel_traits>::Device::compute_kv_tile_idx(
                    params, warpgroup_tile_offset_start, warpgroup_tile_offset_end, kv_steps);

                // Compute the kv_left_mask_end and kv_right_mask_start, where mask is applied when kv_idx <
                // kv_left_mask_end or kv_idx >= kv_right_mask_start.
                auto const [kv_left_mask_end, kv_right_mask_start]
                    = compute_kv_mask_start_end(params, tile_offset_start, tile_offset_end, kv_idx_end);

                // The gmem O tile.
                Gmem_tile_o gmem_o(params, head_info, *shared, tidx, q_step_idx * STEP_Q + local_q_tile_offset);

                // Q ready to use in smem.
                int ready = cbr.peek();
                if (!ready)
                {
                    cbr.wait();
                }

                static_assert(Mma_tile_p::CORES_M == 2);
                float p_max[Mma_tile_p::CORES_M];
                float p_sum[Mma_tile_p::CORES_M];

                int kv_step_idx = kv_idx_start;
                // First K tiles ready to use in smem.
                K_TILE_WAIT();
                // Need to apply mask if only kv tile exists.
                if (kv_idx_start < kv_left_mask_end || kv_idx_start >= kv_right_mask_start)
                {
                    COMPUTE_SINGLE_TILE(true, true);
                }
                else
                {
                    COMPUTE_SINGLE_TILE(true, false);
                }
                KV_TILE_COMPLETE();

                for (kv_step_idx += 1; kv_step_idx < kv_right_mask_start; ++kv_step_idx)
                {

                    // Current step's K tiles ready to use in smem.
                    K_TILE_WAIT();

                    // Move kv tile to next buffer.
                    if (D_GROUPS > 1)
                    {
                        ctile_p.increment_gmma_desc_group();
                    }
                    else
                    {
                        ctile_p.increment_gmma_desc_b_group();
                    }

                    ctile_o.increment_gmma_desc_group();

                    // Apply the start mask only when sliding window attention is enabled.
                    if (kv_step_idx < kv_left_mask_end)
                    {
                        COMPUTE_SINGLE_TILE(false, true);
                    }
                    else
                    {
                        COMPUTE_SINGLE_TILE(false, false);
                    }

                    KV_TILE_COMPLETE();
                }

                // Always apply the mask in the end.
                for (; kv_step_idx < kv_idx_end; ++kv_step_idx)
                {
                    // Current step's K tiles ready to use in smem.
                    K_TILE_WAIT();

                    // Move kv tile to next buffer.
                    if (D_GROUPS > 1)
                    {
                        ctile_p.increment_gmma_desc_group();
                    }
                    else
                    {
                        ctile_p.increment_gmma_desc_b_group();
                    }

                    ctile_o.increment_gmma_desc_group();

                    COMPUTE_SINGLE_TILE(false, true);

                    KV_TILE_COMPLETE();
                }
                if (valid_run)
                {
                    // The per-channel V scale is constant along the BMM2 reduction dimension, so it never has
                    // to interrupt the QGMMA pipeline: it is applied once here, per output column. The
                    // accumulator N index is permuted relative to the head dimension; the column mapping below
                    // mirrors Gmem_tile_o_qgmma_fp32_16bits::store().
                    if constexpr (Kernel_traits::SAGE_V_PER_CHANNEL)
                    {
                        // Layout is (H_kv, D): the amax is reduced over every token of every sequence, so
                        // unlike the Q/K scales it carries no batch dimension.
                        float const* v_scales_ch
                            = params.sage.v.scales + (size_t) (head_info.bidh / params.h_q_per_kv) * params.dv;
                        int const lane_quad = tidx % 4;
#pragma unroll
                        for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
                        {
                            // Each even/odd core pair covers 4 consecutive output columns starting at col_base,
                            // so the scales come in as one 16B load. The (ni, ei) -> column order is exactly
                            // the (x, y, z, w) order of store().
#pragma unroll
                            for (int ni = 0; ni < Mma_tile_o::CORES_N; ni += 2)
                            {
                                int const col_base = mma_ni * Mma_tile_o::CORES_N * 8 + ni * 8 + lane_quad * 4;
                                float4 scale4 = {1.f, 1.f, 1.f, 1.f};
                                if (col_base + 4 <= params.dv)
                                {
                                    scale4 = __ldg(reinterpret_cast<float4 const*>(v_scales_ch + col_base));
                                }
                                float const scale_ch[2][2] = {{scale4.x, scale4.z}, {scale4.y, scale4.w}};
#pragma unroll
                                for (int nj = 0; nj < 2; nj++)
                                {
#pragma unroll
                                    for (int ei = 0; ei < 2; ei++)
                                    {
#pragma unroll
                                        for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
                                        {
                                            ctile_o.acc_[0][mma_ni].elt(
                                                2 * (ni + nj) * Mma_tile_o::CORES_M + 2 * mi + ei)
                                                *= scale_ch[nj][ei];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Final step's update.
                    tile_o_epilogue.scale(ctile_o, p_max, p_sum);
                    // Store o_tile to gmem.
                    gmem_o.store(ctile_o.acc_);
                }

                // Move q, kv to next buffer.
                ctile_p.increment_gmma_desc_a_group();
                ctile_p.increment_gmma_desc_b_group();
                ctile_o.increment_gmma_desc_group();

                if constexpr (Kernel_traits::RETURN_SOFTMAX_STATS)
                {
                    using Mma_tile = typename Traits_p::template Mma_tile<Cta_tile_o>;
                    fmha::Softmax_saver_tma<Cta_tile_o, Mma_tile> saver(params, head_info);
                    saver.store(p_sum, p_max, sqrtf(params.d), q_step_idx * STEP_Q, valid_run);
                }
            }
        }
#ifdef SKIP_SOFTMAX_STAT
        if (tidx == 0)
        {
            atomicAdd(params.skip_softmax_total_blocks, softmax.total_blocks);
            atomicAdd(params.skip_softmax_skipped_blocks, softmax.skipped_blocks);
        }
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool IS_FIRST_COL, bool APPLY_MASK, typename Params>
    inline __device__ void compute_single_tile(Params params, Compute_tile_p& ctile_p, Softmax& softmax,
        Compute_tile_o& ctile_o, float (&p_max)[Mma_tile_p::CORES_M], float (&p_sum)[Mma_tile_p::CORES_M],
        int const tidx, int const actual_kv_seqlen, float const alibi_head_scale, int const row_offset,
        int const col_offset, int const sage_scale_base_kv, Circular_buffer_q_reader& cbr,
        Circular_buffer_kv_reader& cbr_v, OrderedMutexAccessor& mutex, uint32_t* skip_softmax_vote,
        bool complete = false)
    {

        // Skip-softmax vote initialization
        if (tidx == 0)
        {
            // Note that we need a named_barrier_wait in compute_single_tile to make sure init is before voting.
            *skip_softmax_vote = 1;
        }
// load the scales of K/V from global memory
// Load this thread's K scales for one STEP_KV tile. They are stored swizzled as
// [tile][quad_col][chunk], so the SAGE_K_CHUNKS scales a thread needs are contiguous and come in as
// a single 128-bit access.
#define LOAD_SCALES_K(dst)                                                                                             \
    if constexpr (SAGE_BLOCK_SIZE_K > 0)                                                                               \
    {                                                                                                                  \
        const float* _src = params.sage.k.scales + sage_scale_base_kv                                                  \
            + (col_offset / STEP_KV) * SAGE_K_SCALES_PER_TILE + (tidx % 4) * Kernel_traits::SAGE_K_CHUNKS;             \
        if constexpr (Kernel_traits::SAGE_K_CHUNKS == 4)                                                               \
        {                                                                                                              \
            const float4 _s4 = __ldg(reinterpret_cast<const float4*>(_src));                                           \
            dst[0] = _s4.x;                                                                                            \
            dst[1] = _s4.y;                                                                                            \
            dst[2] = _s4.z;                                                                                            \
            dst[3] = _s4.w;                                                                                            \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            _Pragma("unroll") for (int _g = 0; _g < Kernel_traits::SAGE_K_CHUNKS; _g++)                                \
            {                                                                                                          \
                dst[_g] = __ldg(_src + _g);                                                                            \
            }                                                                                                          \
        }                                                                                                              \
    }

        // Load the needed packed masks.
        softmax.load_packed_mask(row_offset, col_offset);

        // experiments show that here is the best place to load scales of K
        float scales_k[Kernel_traits::SAGE_K_CHUNKS];
        LOAD_SCALES_K(scales_k)

        // Wait until another warpgroup has already executed HGMMA.
        if constexpr (ENABLE_MUTEX && Kernel_traits::ELEMENT_BYTES == 2)
        {
            mutex.wait();
        }

        // Ctile_p is only used once by each n step.
        ctile_p.clear();

        // If skip_softmax is enabled, make sure there is no racing between the initialization and writing of
        // skip_softmax_vote.
        named_barrier_wait(Kernel_traits::SKIP_SOFTMAX_BARRIER_ID + threadIdx.x / 128, 128);

        // BMM1 (Q x K').
        warpgroup_arrive();

// Only single K groups when sizeof(D) <= 128B.
#pragma unroll
        for (int kbi = 0; kbi < BMM1_MMAS_K_GROUPS - 1; kbi++)
        {
#pragma unroll
            for (int ki = 0; ki < BMM1_MMAS_K_PER_GROUP; ki++)
            {
                ctile_p.compute(ki, false, ki == BMM1_MMAS_K_PER_GROUP - 1);
            }
            ctile_p.increment_gmma_desc_group();
        }

#pragma unroll
        for (int ki = 0; ki < BMM1_MMAS_K_PER_GROUP - 1; ki++)
        {
            ctile_p.compute(ki);
        }

        ctile_p.compute(BMM1_MMAS_K_PER_GROUP - 1, true, true);

        warpgroup_commit();
        warpgroup_wait<0>();

        // Arrive when the last tile consumes the q tile.
        if (complete)
        {
            cbr.complete(tidx == 0, cbr.ptr());
            cbr.advance();
        }

        if constexpr (ENABLE_MUTEX && Kernel_traits::ELEMENT_BYTES == 2)
        {
            // Notify another warpgroup to execute HGMMA.
            mutex.arrive();
        }
        if constexpr (ENABLE_MUTEX && Kernel_traits::ELEMENT_BYTES == 1)
        {
            // Wait until another warpgroup has already executed QGMMA.
            mutex.named_bar_wait();
        }

        // Fragment p for BMM2 input
        Fragment_p frag_p[Mma_tile_o::MMAS_K];

        // Unpack the elements from bmm1 output to floats.
        softmax.unpack(ctile_p);
        // apply the scales of K before softmax. A chunk is exactly the set of keys sharing one K
        // scale, i.e. one of the SAGE_K_CHUNKS sub-fragments of this thread's key columns.
        //
        // NOTE: the shipped cubin folds this into the exp2f of the softmax instead, which costs a
        // few multiplies per tile rather than one per element. This tree only has to agree on the
        // ABI, the thread shape and the shared-memory budget, so it takes the simple route.
        if constexpr (SAGE_BLOCK_SIZE_K > 0)
        {
            constexpr int ELTS_PER_CHUNK = Mma_tile_p::CORES_N * 2 / Kernel_traits::SAGE_K_CHUNKS;
            static_assert(
                Mma_tile_p::CORES_N * 2 % Kernel_traits::SAGE_K_CHUNKS == 0, "chunk must divide the fragment");
#pragma unroll
            for (int g = 0; g < Kernel_traits::SAGE_K_CHUNKS; g++)
            {
                float const scale_k = scales_k[g];
#pragma unroll
                for (int j = 0; j < ELTS_PER_CHUNK; j++)
                {
#pragma unroll
                    for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
                    {
                        softmax.elt_[mi][g * ELTS_PER_CHUNK + j] *= scale_k;
                    }
                }
            }
        }

        // Apply the alibi and mask.
        softmax.apply_alibi_and_mask<APPLY_MASK>(
            ctile_p, params.alibi_params, alibi_head_scale, actual_kv_seqlen, row_offset, col_offset);

        // Softmax Exp, max/sum, and update scales. If returns false we skip the rest.
        if (!softmax.compute_and_update_scale<IS_FIRST_COL>(p_max, p_sum, skip_softmax_vote))
        {
            if constexpr (ENABLE_MUTEX && Kernel_traits::ELEMENT_BYTES == 1)
            {
                // Notify another warpgroup to execute QGMMA.
                mutex.named_bar_arrive();
            }
            // Need to wait V, otherwise compute-sanitizer synccheck will fail.
            int ready2 = cbr_v.peek();
            if (!ready2)
            {
                cbr_v.wait();
            }

#pragma unroll
            // Advance V descriptor by the same amount as the BMM2 loop would,
            // so that the descriptor stays in sync for subsequent KV steps.
            for (int kbi = 0; kbi < BMM2_MMAS_K_GROUPS - 1; kbi++)
            {
                ctile_o.increment_gmma_desc_group();
            }

            return;
        }

        // Update flash attention scales and pack it for BMM2
        softmax.pack<IS_FIRST_COL>(ctile_o, frag_p);

        if constexpr (ENABLE_MUTEX && Kernel_traits::ELEMENT_BYTES == 1)
        {
            // Notify another warpgroup to execute QGMMA.
            mutex.named_bar_arrive();
        }

        // Wait until v buffer is ready.
        int ready = cbr_v.peek();
        if (!ready)
        {
            cbr_v.wait();
        }

        warpgroup_arrive();

// BMM2 (S * V).
#pragma unroll
        for (int kbi = 0; kbi < BMM2_MMAS_K_GROUPS - 1; kbi++)
        {
#pragma unroll
            for (int ki = 0; ki < BMM2_MMAS_K_PER_GROUP; ++ki)
            {
                int const mma_ki = kbi * BMM2_MMAS_K_PER_GROUP + ki;
                ctile_o.fill_frag_a(frag_p[mma_ki]);
                ctile_o.compute(ki, false, ki == BMM2_MMAS_K_PER_GROUP - 1);
            }
            ctile_o.increment_gmma_desc_group();
        }

#pragma unroll
        for (int ki = 0; ki < BMM2_MMAS_K_PER_GROUP - 1; ++ki)
        {
            int const mma_ki = (BMM2_MMAS_K_GROUPS - 1) * BMM2_MMAS_K_PER_GROUP + ki;
            ctile_o.fill_frag_a(frag_p[mma_ki]);
            ctile_o.compute(ki);
        }

        ctile_o.fill_frag_a(frag_p[Mma_tile_o::MMAS_K - 1]);
        ctile_o.compute(Mma_tile_o::MMAS_K - 1, true, true);

        warpgroup_commit();
        warpgroup_wait<0>();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws
} // namespace fmha
