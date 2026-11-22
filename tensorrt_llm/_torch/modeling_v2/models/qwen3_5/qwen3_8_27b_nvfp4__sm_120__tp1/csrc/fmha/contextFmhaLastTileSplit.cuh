/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Target-private copy of the last q-tile KV split of the SM120 e4m3 head-256 context FMHA.
//
// Namespace TRTLLM_MV2_NS::fmha_sm120::last_tile_split; the softmax and its base-2 helpers are the target's
// (fmha_sm120::Softmax, Softmax_uses_exp2, exp2_update_o, exp2_update_sum, ex2_approx_ftz).
//
// Last q-tile KV split for the tiled no-loop flash attention kernel (device_flash_attention_nl_tiled).
//
// The causal launcher runs one CTA per (q-tile, head, sequence), and every CTA of a long-prefix prefill does about the
// same work. When the grid is a few CTAs over a whole number of waves, those CTAs run as a solo tail wave. The split
// removes that tail:
//   A) the kernel runs q-tiles 0 .. Q-2 (its launch on a params copy with s - STEP), unchanged;
//   B) device_flash_attention_nl_tiled_last_tile_piece runs the last q-tile (Q-1) of every (head, sequence) as
//      num_pieces CTAs, each over a contiguous range of the KV tiles, with the main loop of
//      device_flash_attention_nl_tiled verbatim; it writes the unnormalized FP32 O accumulators and the per-row
//      max / sum to a workspace;
//   C) device_flash_attention_nl_tiled_last_tile_merge combines the pieces in a fixed order (the
//      Tile_o_normalizer::update form: M = max m_j, a_j = expf(m_j - M), acc = sum a_j acc_j, l = sum a_j l_j), then
//      runs the unchanged update_sum / final_update and the register-direct O store.
// Rows of q-tiles 0 .. Q-2 are bitwise those of the unsplit kernel. Rows of the last q-tile are the same math in
// another order (P is quantized against each piece's running max instead of the whole row's).
//
// Only for the configuration with the register-direct O epilogue (Kernel_traits::STORE_O_FROM_REGS: Ada QMMA e4m3
// with FP32 accumulators, 16-bit output, 64 x 256 O tile on 4x1x1 warps) and the plain causal mask.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/fmhaSm120KernelTraits.cuh"
#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/fmhaSm120Softmax.cuh"

#include <fmha/gemm.h>
#include <fmha/utils.h>
#include <fused_multihead_attention_kernel.h>

#include <algorithm>
#include <cfloat>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace fmha_sm120
{
namespace last_tile_split
{

// The unqualified stock names in the copy below (Single_cta, ...) are the fused_multihead_attention ones.
using namespace ::fused_multihead_attention;

////////////////////////////////////////////////////////////////////////////////////////////////////

// The workspace of one piece, in per-thread fragment order (coalesced over the threads of the CTA):
//   acc:   ACC_FLOAT4S_PER_THREAD float4 per thread at [piece][i][tidx];
//   stats: one float4 per thread at [piece][tidx] = (max row 0, max row 1, sum row 0, sum row 1).
// The piece index is (bidb * h + bidh) * num_pieces + j.
template <typename Kernel_traits>
struct Workspace
{
    using Traits_o = typename Kernel_traits::Traits_o;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;
    using Fragment_accu = fmha::Fragment_accumulator<Traits_o>;

    static constexpr int MMAS_M = Mma_tile_o::MMAS_M;
    static constexpr int MMAS_N = Mma_tile_o::VALID_MMAS_N;
    static constexpr int THREADS = Kernel_traits::THREADS;

    static_assert(Fragment_accu::NUM_ELTS == 8, "");
    static_assert(MMAS_M == 1, "Two rows per thread");

    // float4 per thread for the accumulators.
    static constexpr int ACC_FLOAT4S_PER_THREAD = MMAS_M * MMAS_N * Fragment_accu::NUM_ELTS / 4;
    // float4 per piece.
    static constexpr int ACC_FLOAT4S_PER_PIECE = ACC_FLOAT4S_PER_THREAD * THREADS;
    static constexpr int STATS_FLOAT4S_PER_PIECE = THREADS;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The pieces actually used for a last q-tile with num_kv_tiles KV tiles: at most num_pieces, and at least 2 tiles
// per piece, so the last piece always starts at or before the first masked tile (the causal mask of a 64-row q-tile
// touches at most the last 2 tiles of 128 keys) and no row of any piece is fully masked.
inline __device__ int active_pieces(int num_kv_tiles, int num_pieces)
{
    int const by_tiles = num_kv_tiles / 2;
    return max(1, min(num_pieces, by_tiles));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// B) One piece of the last q-tile. Grid (num_pieces, h, b). The body is the main loop of
// device_flash_attention_nl_tiled (fused_multihead_flash_attention_kernel_noloop_tiled.h), with q_loop fixed and the
// kv loop limited to the piece's range.
template <typename Kernel_traits, typename Params>
inline __device__ void device_flash_attention_nl_tiled_last_tile_piece(
    Params const& params, int const q_loop, int const num_pieces, float4* ws_acc, float4* ws_stats)
{

    // The instruction traits.
    using Traits_p = typename Kernel_traits::Traits_p;
    using Traits_o = typename Kernel_traits::Traits_o;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    using Ws = Workspace<Kernel_traits>;

    // Do we use LDGSTS for Q, K or V?
    static constexpr int USE_LDGSTS_Q = Kernel_traits::USE_LDGSTS_Q;

    static constexpr int USE_LDGSTS_K = Kernel_traits::USE_LDGSTS_K;

    static constexpr int USE_LDGSTS_V = Kernel_traits::USE_LDGSTS_V;

    // Do we use LDGSTS for any of the 3 input matrices.
    static constexpr int USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V;

    static_assert(USE_LDGSTS, "Supports only USE_LDGSTS = true");
    static_assert(!Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");
    static_assert(Kernel_traits::STORE_O_FROM_REGS, "Only the register-direct O configuration");
    static_assert(Kernel_traits::CAUSAL_MASK && !Kernel_traits::SLIDING_WINDOW_ATTENTION
            && !Kernel_traits::BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION && !Kernel_traits::CUSTOM_MASK
            && !Kernel_traits::IS_MTP,
        "Only the plain causal mask");
    static_assert(Cta_tile_o::VALID_N <= Cta_tile_o::N, "One CTA per O row");

    // Do we need to check if there are negative inf for softmax row_max ?
    static constexpr int CHECK_NEG_INF = Kernel_traits::SLIDING_WINDOW_ATTENTION || Kernel_traits::CUSTOM_MASK;

    // Shared memory.
    extern __shared__ char smem_[];

    // The piece of the KV range.
    int const piece = blockIdx.x;
    // The block index for the batch.
    int const bidb = blockIdx.z;
    // The block index for the head.
    int const bidh = blockIdx.y;
    // The thread index.
    int const tidx = threadIdx.x;

    // The block info.
    Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
    // The local sequence offset of Q.
    int q_sequence_start = q_loop * Gmem_tile_q::ROWS;
    // Consider the past sequence length.
    q_sequence_start += binfo.actual_kv_seqlen - binfo.actual_q_seqlen;
    if (binfo.stop_early(q_loop * Gmem_tile_q::ROWS))
    {
        return;
    }

    // The KV range of the whole q-tile, as in device_flash_attention_nl_tiled (causal, no sliding window).
    int const valid_seqlen = min(q_sequence_start + Cta_tile_p::M, binfo.actual_kv_seqlen);
    int const num_kv_tiles = fmha::div_up(valid_seqlen, int(Cta_tile_p::N));
    int const active = active_pieces(num_kv_tiles, num_pieces);
    if (piece >= active)
    {
        return;
    }
    // The range of this piece: tiles [piece * n / active, (piece + 1) * n / active).
    int const kv_loop_start = (piece * num_kv_tiles / active) * int(Cta_tile_p::N);
    int const kv_loop_end = ((piece + 1) * num_kv_tiles / active) * int(Cta_tile_p::N);

    // Create the object to control the masks.
    fmha::Mask_dispatcher<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION, Kernel_traits::IS_MTP> mask(
        params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx, q_loop * Gmem_tile_q::ROWS);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx, 0, 0);
    // Allocate the shared memory tile loader for V.
    Smem_tile_v smem_v(&smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], tidx);

    // With chunked attention, the q_start_seqlen might not be multiple of Cta_tile_p::M.
    int const kv_mask_loop_start = int(q_sequence_start / Cta_tile_p::N) * Cta_tile_p::N;

    // Move K and V tiles to the start of the piece.
    gmem_k.move_by_offset(kv_loop_start);
    gmem_v.move_by_offset(kv_loop_start);

    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);

    // Push the LDGDEPBAR instruction after the loads for Q, K
    fmha::ldgdepbar<USE_LDGSTS>();

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_k.commit(smem_k);

    // Declare the fragments for Q/K/V.
    typename Smem_tile_q::Fragment frag_q[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
    typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::VALID_MMAS_N];

    // Create the object to do the softmax.
    using Softmax = fmha_sm120::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, &smem_[Smem_tile_q::BYTES_PER_TILE], bidb, tidx);

    static_assert(!Softmax::USE_SHARED_MEMORY, "");
    static_assert(Softmax::ROWS_PER_THREAD == 2, "");

    // The base-2 softmax (Kernel_traits::USE_EXP2_SOFTMAX, see fmhaSm120Softmax.cuh): the stats below hold the row max
    // in raw units, which the merge kernel reads in the same units.
    static constexpr bool USE_EXP2 = fmha_sm120::Softmax_uses_exp2<Softmax>::VALUE;

    // Load the mask for that iteration.
    mask.load(q_sequence_start);

    // Declare the accumulators for the 2nd gemm.
    fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
    using Acc_type_o = typename Traits_o::Accumulator_type;
    fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

    // Flash attention updater
    fmha::Tile_o_normalizer<Traits_o, Cta_tile_o> acc_o_normalizer(params, binfo);
    float global_max[Softmax::ROWS_PER_THREAD];
    float global_sum[Softmax::ROWS_PER_THREAD];

    constexpr int BMM1_VALID_MMAS_K = Mma_tile_p::VALID_MMAS_K;
    constexpr int BMM1_TAIL_MMAS_K_BOUND
        = BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K ? BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K : Mma_tile_p::MMAS_K;
    constexpr int BMM1_MAIN_MMAS_K_BOUND = BMM1_VALID_MMAS_K - BMM1_TAIL_MMAS_K_BOUND;

    constexpr int BMM2_TAIL_MMAS_K_BOUND = Mma_tile_o::MMAS_K;
    constexpr int BMM2_MAIN_MMAS_K_BOUND = Kernel_traits::TOTAL_BMM2_MMAS_K - BMM2_TAIL_MMAS_K_BOUND;

    for (int kv_loop = kv_loop_start; kv_loop < kv_loop_end; kv_loop += Cta_tile_p::N)
    {

        bool const first_step = (kv_loop == kv_loop_start);
        bool const apply_mask = params.has_alibi || (kv_loop >= kv_mask_loop_start);

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        using Acc_type_p = typename Traits_p::Accumulator_type;
        fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Move mask offset.
        mask.move_to_offset(kv_loop);

        // BMM1 main loop
        for (int bmm1_k = 0; bmm1_k < BMM1_MAIN_MMAS_K_BOUND; bmm1_k += Mma_tile_p::MMAS_K)
        {

            // Trigger the load for the next Q/K values
            if (Kernel_traits::RELOAD_Q || first_step)
            {
                gmem_q.move_col();
                smem_q.move_to_next_write_buffer();
                gmem_q.load(smem_q);
            }

            gmem_k.move_col();
            smem_k.move_to_next_write_buffer();
            gmem_k.load(smem_k);

            // Push the LDGDEPBAR instruction after the loads for QK.
            fmha::ldgdepbar<USE_LDGSTS>();

            if (Kernel_traits::RELOAD_Q || first_step)
            {
                gmem_q.commit(smem_q);
            }
            gmem_k.commit(smem_k);

            // Double SMEM buffer: make sure we are done writing the data from last stage (smem).
            // Leave 1 outstanding batch loadings
            fmha::depbar_<USE_LDGSTS, 1>();
            __syncthreads();

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
            for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                // Load the fragments for Q/K inner tile
                smem_q.load(frag_q[ki], ki);
                smem_k.load(frag_k[ki], ki);

                // Do the math for the values already in registers.
                fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
            }

            // Make sure we are done reading the data (smem).
            __syncthreads();

            smem_q.move_to_next_read_buffer();
            smem_k.move_to_next_read_buffer();
        }

        // BMM1 tail loop
        {
            // Trigger the load for next V values
            if (kv_loop > kv_loop_start)
            {
                gmem_v.move();
                smem_v.move_to_next_write_buffer();
            }
            gmem_v.load(smem_v);

            fmha::ldgdepbar<USE_LDGSTS>();

            gmem_v.commit(smem_v);

            // Wait for Q/K to finish loading
            fmha::depbar_<USE_LDGSTS, 1>();
            __syncthreads();

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
            for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                // Load the fragments for Q/K inner tile
                smem_q.load(frag_q[ki], ki);
                smem_k.load(frag_k[ki], ki);

                // Do the math for the values already in registers.
                if (Cta_tile_p::VALID_K % Cta_tile_p::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND)
                {
                    fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
                }
            }
            smem_q.move_to_next_read_buffer();
            smem_k.move_to_next_read_buffer();
        } // end BMM1 tail loop

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack(acc_p);

        // Apply the mask.
        if (apply_mask)
        {
            if (params.has_alibi)
            {
                softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
            }
            else
            {
                softmax.apply_mask(mask);
            }
        }

        // First step of the flash attention
        if (first_step)
        {
            // Compute the max.
            softmax.template reduce<fmha::Max_>(global_max);

            softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);

            // Compute the sum.
            softmax.template reduce<fmha::Sum_>(global_sum);
        }
        else
        {
            float tmp[Softmax::ROWS_PER_THREAD];

#pragma unroll
            for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
            {
                tmp[i] = global_max[i];
            }

            // Compute the max.
            softmax.template reduce<fmha::Max_>(global_max);

            // Update last step's acc_o.
            if constexpr (USE_EXP2)
            {
                fmha_sm120::exp2_update_o(acc_o, global_max, tmp, global_sum, softmax.s2_);
            }
            else
            {
                acc_o_normalizer.update(acc_o, global_max, tmp, global_sum);
            }
            // Apply expf of softmax.
            softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);

#pragma unroll
            for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
            {
                tmp[i] = global_sum[i];
                global_sum[i] = 0.f;
            }

            // Compute the sum.
            softmax.template reduce<fmha::Sum_>(global_sum);

#pragma unroll
            for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
            {
                global_sum[i] += tmp[i];
            }
        }

        // Repack for the next BMM.
        fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Kernel_traits::TOTAL_BMM2_MMAS_K][Mma_tile_o::MMAS_M];
        softmax.pack(frag_p);

        // BMM2 main loop
        for (int bmm2_k = 0; bmm2_k < BMM2_MAIN_MMAS_K_BOUND; bmm2_k += Mma_tile_o::MMAS_K)
        {
            // Trigger the load for next V values
            gmem_v.move();
            smem_v.move_to_next_write_buffer();
            gmem_v.load(smem_v);

            // Push the LDGDEPBAR instruction after the loads for K.
            fmha::ldgdepbar<USE_LDGSTS>();
            gmem_v.commit(smem_v);

            // Leave 1 outstanding batch loadings
            fmha::depbar_<USE_LDGSTS, 1>();
            __syncthreads();

#pragma unroll
            for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
            {
                int p_ki = bmm2_k + ki;
                smem_v.load(frag_v[ki], ki);
                fmha::gemm(acc_o, frag_p[p_ki], frag_v[ki]);
            }

            // Make sure we are done reading the data (smem).
            __syncthreads();

            smem_v.move_to_next_read_buffer();
        }

        // BMM2 tail loop
        {
            if (kv_loop + Cta_tile_p::N < kv_loop_end)
            {

                // Trigger the load for the next Q/K values
                // Advance K rows and rewind QK cols
                int col_steps = fmha::div_up((int) Kernel_traits::VALID_D, (int) Kernel_traits::CTA_P_TILE_K);
                // Enforce warp-uniform value.
                col_steps = __shfl_sync(0xffffffff, col_steps, 0);

                if constexpr (Kernel_traits::RELOAD_Q)
                {
                    gmem_q.move_col();
                    gmem_q.rewind_col(col_steps);
                    smem_q.move_to_next_write_buffer();
                    gmem_q.load(smem_q);
                }
                gmem_k.move();
                gmem_k.move_col();
                gmem_k.rewind_col(col_steps);
                smem_k.move_to_next_write_buffer();
                gmem_k.load(smem_k);

                // Push the LDGDEPBAR instruction after the loads for K.
                fmha::ldgdepbar<USE_LDGSTS>();

                if constexpr (Kernel_traits::RELOAD_Q)
                {
                    gmem_q.commit(smem_q);
                }
                gmem_k.commit(smem_k);

                // Wait for V to finish loading
                fmha::depbar_<USE_LDGSTS, 1>();
                __syncthreads();
            }
            else
            {
                fmha::depbar_<USE_LDGSTS, 0>();
                __syncthreads();
            }

// Do this part of O = P^T * V^T.
#pragma unroll
            for (int ki = 0; ki < BMM2_TAIL_MMAS_K_BOUND; ++ki)
            {
                int p_ki = BMM2_MAIN_MMAS_K_BOUND + ki;
                smem_v.load(frag_v[ki], ki);
                fmha::gemm(acc_o, frag_p[p_ki], frag_v[ki]);
            }

            smem_v.move_to_next_read_buffer();
        } // end BMM2 tail loop

    }     // Inner loop over the key/value sequence length.

    // Make sure there is no data in-flight.
    fmha::depbar_<USE_LDGSTS, 0>();

    if constexpr (USE_EXP2)
    {
        // The base-2 row sums carry scale_softmax (folded into the exponent): back to the expf path's units.
        float const to_base = softmax.sum_to_base_units();
        global_sum[0] *= to_base;
        global_sum[1] *= to_base;
    }

    // Write the unnormalized accumulators and the row max / sum of the piece.
    size_t const piece_idx = (size_t(bidb) * params.h + bidh) * num_pieces + piece;
    float4* acc_ptr = ws_acc + piece_idx * Ws::ACC_FLOAT4S_PER_PIECE + tidx;
#pragma unroll
    for (int mi = 0; mi < Ws::MMAS_M; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < Ws::MMAS_N; ++ni)
        {
#pragma unroll
            for (int hi = 0; hi < 2; ++hi)
            {
                int const i = (mi * Ws::MMAS_N + ni) * 2 + hi;
                acc_ptr[i * Ws::THREADS] = make_float4(acc_o[mi][ni].elt(4 * hi + 0), acc_o[mi][ni].elt(4 * hi + 1),
                    acc_o[mi][ni].elt(4 * hi + 2), acc_o[mi][ni].elt(4 * hi + 3));
            }
        }
    }
    ws_stats[piece_idx * Ws::STATS_FLOAT4S_PER_PIECE + tidx]
        = make_float4(global_max[0], global_max[1], global_sum[0], global_sum[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// C) The merge of the pieces of the last q-tile. Grid (h, b), Kernel_traits::THREADS threads, no shared memory.
template <typename Kernel_traits, typename Params>
inline __device__ void device_flash_attention_nl_tiled_last_tile_merge(
    Params const& params, int const q_loop, int const num_pieces, float4 const* ws_acc, float4 const* ws_stats)
{
    using Traits_o = typename Kernel_traits::Traits_o;
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_o_store = typename Kernel_traits::Gmem_tile_o_from_regs;
    using Ws = Workspace<Kernel_traits>;

    static_assert(Kernel_traits::STORE_O_FROM_REGS, "Only the register-direct O configuration");

    // The block index for the head.
    int const bidh = blockIdx.x;
    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The thread index.
    int const tidx = threadIdx.x;

    // The block info.
    Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
    int q_sequence_start = q_loop * Gmem_tile_q::ROWS;
    q_sequence_start += binfo.actual_kv_seqlen - binfo.actual_q_seqlen;
    if (binfo.stop_early(q_loop * Gmem_tile_q::ROWS))
    {
        return;
    }

    // The pieces written by the piece kernel (the same arithmetic).
    int const valid_seqlen = min(q_sequence_start + Cta_tile_p::M, binfo.actual_kv_seqlen);
    int const num_kv_tiles = fmha::div_up(valid_seqlen, int(Cta_tile_p::N));
    int const active = active_pieces(num_kv_tiles, num_pieces);

    size_t const piece0 = (size_t(bidb) * params.h + bidh) * num_pieces;

    // The row max over the pieces.
    float global_max[2] = {-FLT_MAX, -FLT_MAX};
    for (int j = 0; j < active; ++j)
    {
        float4 const s = ws_stats[(piece0 + j) * Ws::STATS_FLOAT4S_PER_PIECE + tidx];
        global_max[0] = fmax(global_max[0], s.x);
        global_max[1] = fmax(global_max[1], s.y);
    }

    // The units of the piece stats: raw accumulator units on the base-2 path (see fmhaSm120Softmax.cuh).
    using Softmax = fmha_sm120::Softmax<typename Kernel_traits::Traits_p, Cta_tile_p, Kernel_traits>;
    static constexpr bool USE_EXP2 = fmha_sm120::Softmax_uses_exp2<Softmax>::VALUE;
    uint32_t const scale_bmm1_u = params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1;
    float const scale_bmm1 = reinterpret_cast<float const&>(scale_bmm1_u);
    float const s2 = scale_bmm1 * 1.4426950408889634f;

    // The weighted sums, in the fixed order j = 0 .. active - 1.
    fmha::Fragment_accumulator<Traits_o> acc_o[Ws::MMAS_M][Ws::MMAS_N];
    using Acc_type_o = typename Traits_o::Accumulator_type;
    fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);
    float global_sum[2] = {0.f, 0.f};
    for (int j = 0; j < active; ++j)
    {
        float4 const s = ws_stats[(piece0 + j) * Ws::STATS_FLOAT4S_PER_PIECE + tidx];
        float alpha[2];
        if constexpr (USE_EXP2)
        {
            alpha[0] = fmha_sm120::ex2_approx_ftz((s.x - global_max[0]) * s2);
            alpha[1] = fmha_sm120::ex2_approx_ftz((s.y - global_max[1]) * s2);
        }
        else
        {
            alpha[0] = expf(s.x - global_max[0]);
            alpha[1] = expf(s.y - global_max[1]);
        }
        global_sum[0] += alpha[0] * s.z;
        global_sum[1] += alpha[1] * s.w;
        float4 const* acc_ptr = ws_acc + (piece0 + j) * Ws::ACC_FLOAT4S_PER_PIECE + tidx;
#pragma unroll
        for (int mi = 0; mi < Ws::MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < Ws::MMAS_N; ++ni)
            {
#pragma unroll
                for (int hi = 0; hi < 2; ++hi)
                {
                    int const i = (mi * Ws::MMAS_N + ni) * 2 + hi;
                    float4 const v = acc_ptr[i * Ws::THREADS];
                    // Elements {0, 1, 4, 5} belong to row 0 and {2, 3, 6, 7} to row 1 (Tile_o_normalizer_fp32).
                    acc_o[mi][ni].elt(4 * hi + 0) += alpha[0] * v.x;
                    acc_o[mi][ni].elt(4 * hi + 1) += alpha[0] * v.y;
                    acc_o[mi][ni].elt(4 * hi + 2) += alpha[1] * v.z;
                    acc_o[mi][ni].elt(4 * hi + 3) += alpha[1] * v.w;
                }
            }
        }
    }

    // The unchanged epilogue of device_flash_attention_nl_tiled.
    fmha::Tile_o_normalizer<Traits_o, Cta_tile_o> acc_o_normalizer(params, binfo);
    // Update the sum if attention sinks are used.
    if constexpr (USE_EXP2)
    {
        fmha_sm120::exp2_update_sum(acc_o_normalizer.attention_sink_value_, global_max, global_sum, scale_bmm1,
            Kernel_traits::Traits_p::SOFTMAX_FP_QUANT_SCALE);
    }
    else
    {
        acc_o_normalizer.update_sum(global_max, global_sum);
    }
    // Update acc_o of flash attention
    acc_o_normalizer.final_update(acc_o, global_sum);

    Gmem_tile_o_store gmem_o(params, binfo, tidx, q_loop * Gmem_tile_o::ROWS, 0);
    gmem_o.store(acc_o);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
__global__ void fmha_v2_nl_tiled_last_tile_piece_kernel(
    bert::Fused_multihead_attention_params_v2 params, int q_loop, int num_pieces, float4* ws_acc, float4* ws_stats)
{
    device_flash_attention_nl_tiled_last_tile_piece<Kernel_traits>(params, q_loop, num_pieces, ws_acc, ws_stats);
}

template <typename Kernel_traits>
__global__ void fmha_v2_nl_tiled_last_tile_merge_kernel(bert::Fused_multihead_attention_params_v2 params, int q_loop,
    int num_pieces, float4 const* ws_acc, float4 const* ws_stats)
{
    device_flash_attention_nl_tiled_last_tile_merge<Kernel_traits>(params, q_loop, num_pieces, ws_acc, ws_stats);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Host side.

// The number of pieces for the last q-tile (0: do not split). slots = SMs x CTAs per SM of the kernel. The split runs
// for at most max_batch sequences, when dropping the last q-tile from the kernel's grid saves a whole wave; the pieces
// of all (head, sequence) pairs then fill at most one wave.
inline int choose_num_pieces(
    int b, int h, int s, int step_q, int slots, int max_pieces_total, int max_pieces, int max_batch)
{
    if (b <= 0 || b > max_batch || h <= 0 || slots <= 0 || s <= 0)
    {
        return 0;
    }
    long long const q_tiles = (s + step_q - 1) / step_q;
    if (q_tiles < 2)
    {
        return 0;
    }
    long long const ctas = q_tiles * h * b;
    long long const ctas_a = (q_tiles - 1) * h * b;
    if ((ctas_a + slots - 1) / slots >= (ctas + slots - 1) / slots)
    {
        return 0;
    }
    long long k = slots / ((long long) h * b);
    k = std::min<long long>(k, max_pieces);
    k = std::min<long long>(k, max_pieces_total / ((long long) h * b));
    return k >= 2 ? int(k) : 0;
}

// Launches B and C (the caller runs A: the kernel's launch with s - step_q). Returns the launch error.
template <typename Kernel_traits>
inline cudaError_t launch_last_tile_split(bert::Fused_multihead_attention_params_v2 const& params, int q_loop,
    int num_pieces, float4* ws_acc, float4* ws_stats, cudaStream_t stream)
{
    constexpr int smem = Kernel_traits::BYTES_PER_SMEM;
    if (smem >= 48 * 1024)
    {
        cudaError_t err = cudaFuncSetAttribute(
            fmha_v2_nl_tiled_last_tile_piece_kernel<Kernel_traits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    dim3 const grid_b(num_pieces, params.h, params.b);
    fmha_v2_nl_tiled_last_tile_piece_kernel<Kernel_traits>
        <<<grid_b, Kernel_traits::THREADS, smem, stream>>>(params, q_loop, num_pieces, ws_acc, ws_stats);
    dim3 const grid_c(params.h, params.b);
    fmha_v2_nl_tiled_last_tile_merge_kernel<Kernel_traits>
        <<<grid_c, Kernel_traits::THREADS, 0, stream>>>(params, q_loop, num_pieces, ws_acc, ws_stats);
    return cudaGetLastError();
}

} // namespace last_tile_split
} // namespace fmha_sm120

TRTLLM_MV2_NAMESPACE_END
