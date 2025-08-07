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

    // sanitize 0 to -1, avoid DIV BY ZERO below
    enum
    {
        SAGE_BLOCK_SIZE_K = Kernel_traits::SAGE_BLOCK_SIZE_K > 0 ? Kernel_traits::SAGE_BLOCK_SIZE_K : -1
    };

    enum
    {
        SAGE_BLOCK_SIZE_V = Kernel_traits::SAGE_BLOCK_SIZE_V > 0 ? Kernel_traits::SAGE_BLOCK_SIZE_V : -1
    };

    // BLOCK_SIZE_Q should be multiply of STEP_Q (usually 64) so that q scale can be fused into scale_bmm1
    static_assert(SAGE_BLOCK_SIZE_Q < 0 || SAGE_BLOCK_SIZE_Q % STEP_Q == 0);
    static_assert(SAGE_BLOCK_SIZE_K < 0 || SAGE_BLOCK_SIZE_K % 8 == 0);  // 8 = columns of a gmma CORE
    static_assert(SAGE_BLOCK_SIZE_V < 0 || SAGE_BLOCK_SIZE_V % 32 == 0); // 32 = K dimension of a qgmma

    // SAGE_BLOCKS_PER_STEP_X is used to declare scale buffer like `float scales_k[SAGE_BLOCKS_PER_STEP_K];`
    // if SAGE_BLOCKS_PER_STEP_X == 0, you will get `zero-sized variable is not allowed in device code`
    // error from nvcc, so the minimal value have to be 1. But don't worry, unused local variables will
    // be optimized out by compiler.
    enum
    {
        SAGE_BLOCKS_PER_STEP_K = std::max(STEP_KV / SAGE_BLOCK_SIZE_K, 1)
    };

    enum
    {
        SAGE_BLOCKS_PER_STEP_V = std::max(STEP_KV / SAGE_BLOCK_SIZE_V, 1)
    };

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
        kv_step_idx * STEP_KV, sage_scale_row, cbr, cbr_v, mutex_accessor, kv_step_idx == kv_idx_end - 1);

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

        // The left mask is needed when we attend to a specific sliding window or chunk.
        if constexpr (SLIDING_OR_CHUNKED_ATTENTION)
        {
            // The kv_left_mask_end is the start of the chunk.
            kv_left_mask_end = div_up(is_chunked_attention
                    ? ((tile_offset_end >> params.log2_chunked_attention_size) << params.log2_chunked_attention_size)
                    : (tile_offset_end + 1 - params.sliding_window_size),
                STEP_KV);
        }

        // The right mask is needed when causal mask (including sliding_window_attention or chunked attention) is used.
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

            // Calculate the alibi head_scaling_factor.
            float alibi_head_scale
                = APPLY_ALIBI ? get_alibi_head_scaling_factor<AlibiParams>(head_info.bidh, params.alibi_params) : 0.f;
            // pre-compute the row of the scale for reuse
            int sage_scale_row;
            if constexpr (Kernel_traits::SAGE_ATTENTION)
            {
                sage_scale_row = head_info.bidb * params.h + head_info.bidh;
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
                    int const idx = sage_scale_row * params.sage.q.max_nblock + q_offset / SAGE_BLOCK_SIZE_Q;
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
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool IS_FIRST_COL, bool APPLY_MASK, typename Params>
    inline __device__ void compute_single_tile(Params params, Compute_tile_p& ctile_p, Softmax& softmax,
        Compute_tile_o& ctile_o, float (&p_max)[Mma_tile_p::CORES_M], float (&p_sum)[Mma_tile_p::CORES_M],
        int const tidx, int const actual_kv_seqlen, float const alibi_head_scale, int const row_offset,
        int const col_offset, int const sage_scale_row, Circular_buffer_q_reader& cbr, Circular_buffer_kv_reader& cbr_v,
        OrderedMutexAccessor& mutex, bool complete = false)
    {
// load the scales of K/V from global memory
#define LOAD_SCALES_KV(dst, which, blocks_per_step, block_size)                                                        \
    if constexpr (block_size > 0)                                                                                      \
    {                                                                                                                  \
        const int _start = col_offset / block_size;                                                                    \
        const float* _src = params.sage.which.scales + sage_scale_row * params.sage.which.max_nblock + _start;         \
        const int _end = params.sage.which.max_nblock - _start;                                                        \
        _Pragma("unroll") for (int _i = 0; _i < blocks_per_step; _i++)                                                 \
        {                                                                                                              \
            dst[_i] = _i < _end ? _src[_i] : 1.0f;                                                                     \
        }                                                                                                              \
    }

#define LOAD_SCALES_K(scales) LOAD_SCALES_KV(scales, k, SAGE_BLOCKS_PER_STEP_K, SAGE_BLOCK_SIZE_K)

#define LOAD_SCALES_V(scales) LOAD_SCALES_KV(scales, v, SAGE_BLOCKS_PER_STEP_V, SAGE_BLOCK_SIZE_V)

        // Load the needed packed masks.
        softmax.load_packed_mask(row_offset, col_offset);

        // experiments show that here is the best place to load scales of K
        float scales_k[SAGE_BLOCKS_PER_STEP_K];
        LOAD_SCALES_K(scales_k)

        // Wait until another warpgroup has already executed HGMMA.
        if constexpr (ENABLE_MUTEX && Kernel_traits::ELEMENT_BYTES == 2)
        {
            mutex.wait();
        }

        // Ctile_p is only used once by each n step.
        ctile_p.clear();

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
        // apply the scales of K before softmax
        if constexpr (SAGE_BLOCK_SIZE_K > 0)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
                float const scale_k = scales_k[SAGE_BLOCKS_PER_STEP_K * ni / Mma_tile_p::CORES_N];
#pragma unroll
                for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
                {
                    softmax.elt_[mi][2 * ni] *= scale_k;
                    softmax.elt_[mi][2 * ni + 1] *= scale_k;
                }
            }
        }

        // Apply the alibi and mask.
        softmax.apply_alibi_and_mask<APPLY_MASK>(
            ctile_p, params.alibi_params, alibi_head_scale, actual_kv_seqlen, row_offset, col_offset);

        // Softmax Exp, max/sum, and update scales.
        softmax.compute_and_update_scale<IS_FIRST_COL>(p_max, p_sum);

        // experiments show that here is the best place to load scales of V
        float scales_v[SAGE_BLOCKS_PER_STEP_V];
        LOAD_SCALES_V(scales_v)

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

        float last_scale_v;

// Apply the scale of V to partial result.
// Note 2 points:
// 1. Because the matrix V is quantized along the inner dimension, it is necessary to interrupt
//   the MMA workflow after processing each BLOCKS_SIZE_V rows of V and scale the intermediate
//   results once. For example, STEP_KV=256, qgmma.K=32, then 256/32=8 MMAs are needs,
//   so mma_ki = [0,1,2, ..., 7]. If the BLOCK_SIZE_V=64, then after each 2 qgmmas we should scale
//   ctile_o.
// 2. The ctile_o is all zero at the beginning. if we directly apply the scale of V after each 2
//   qgmmas, let's see what happens:
//     ctile_o = [0]
//     ctile_o = (ctile_o + P0 x V0) * s0 = P0 x V0 * s0
//     ctile_o = (ctile_o + P1 x V1) * s1 = P0 x V0 * s0 * s1 + P1 x V1 * s1
//     ctile_o = (ctile_o + P2 x V2) * s2 = P0 x V0 * s0 * s1 * s2 + P1 x V1 * s1 * s2 + P2 x V2 * s2
//     ...
//   As you see, the actual scale of a V block is the cumulative product of the scales of all
//   later blocks. To solve this, we have to preprocess the scale s[i] of block[i] to s[i]/s[i+1],
//   and the final block uses the actual scale.
// But to fetch the next scale in next STEP leads to bad performance. So we apply s[i-1]/s[i] to
// current partial result BEFORE each V block.
#define APPLY_SCALE_V(mma_ki)                                                                                          \
    if constexpr (SAGE_BLOCK_SIZE_V > 0)                                                                               \
    {                                                                                                                  \
        if (mma_ki % (Mma_tile_o::MMAS_K / SAGE_BLOCKS_PER_STEP_V) == 0)                                               \
        {                                                                                                              \
            float _scale_v = scales_v[SAGE_BLOCKS_PER_STEP_V * mma_ki / Mma_tile_o::MMAS_K];                           \
            if (mma_ki != 0)                                                                                           \
            {                                                                                                          \
                warpgroup_commit();                                                                                    \
                warpgroup_wait<0>();                                                                                   \
            }                                                                                                          \
            last_scale_v = _scale_v;                                                                                   \
        }                                                                                                              \
    }

// BMM2 (S * V).
#pragma unroll
        for (int kbi = 0; kbi < BMM2_MMAS_K_GROUPS - 1; kbi++)
        {
#pragma unroll
            for (int ki = 0; ki < BMM2_MMAS_K_PER_GROUP; ++ki)
            {
                int const mma_ki = kbi * BMM2_MMAS_K_PER_GROUP + ki;
                APPLY_SCALE_V(mma_ki)
                ctile_o.fill_frag_a(frag_p[mma_ki]);
                ctile_o.compute(ki, false, ki == BMM2_MMAS_K_PER_GROUP - 1);
            }
            ctile_o.increment_gmma_desc_group();
        }

#pragma unroll
        for (int ki = 0; ki < BMM2_MMAS_K_PER_GROUP - 1; ++ki)
        {
            int const mma_ki = (BMM2_MMAS_K_GROUPS - 1) * BMM2_MMAS_K_PER_GROUP + ki;
            APPLY_SCALE_V(mma_ki)
            ctile_o.fill_frag_a(frag_p[mma_ki]);
            ctile_o.compute(ki);
        }

        APPLY_SCALE_V((Mma_tile_o::MMAS_K - 1))
        ctile_o.fill_frag_a(frag_p[Mma_tile_o::MMAS_K - 1]);
        ctile_o.compute(Mma_tile_o::MMAS_K - 1, true, true);

        warpgroup_commit();
        warpgroup_wait<0>();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws
} // namespace fmha
