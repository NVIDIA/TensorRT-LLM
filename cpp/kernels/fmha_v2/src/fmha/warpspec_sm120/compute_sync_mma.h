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

#pragma once

// skip_softmax consumer: BMM1 + softmax + skip-softmax + BMM2 body.
//
// This is a port of fused_multihead_flash_attention_kernel_noloop_tiled.h
// where:
//
//   * `gmem_q.load(smem_q)` / `gmem_k.load(smem_k)` / `gmem_v.load(smem_v)`
//     are removed -- the producer warp (in dma_sync_mma.h) issues TMA loads
//     into the ring instead.
//   * `fmha::ldgdepbar<USE_LDGSTS>()` + `__syncthreads()` between load and
//     compute are replaced by `cbr_*.wait()` against the entry-produced
//     mbarrier for the slot we're about to read.
//   * After consumer is done with a slot, `cbr_*.complete(tidx == 0, slot)`
//     arrives on the entry-consumed mbarrier so the producer can recycle.
//
// EVERYTHING ELSE -- BMM1 inner loop via fmha::gemm(), softmax, mask, the
// per-warp skip-softmax vote with log-threshold, the BMM2 split between
// skip-path and no-skip-path -- mirrors the non-warp-specialized tiled sm_120
// kernel so the attention math is numerically identical.

#include <fmha/gemm.h>
#include <fmha/mask.h>
#include <fmha/softmax.h>
#include <fmha/utils.h>
#include <fmha/warpspec/circular_buffer.h>
#include <fused_multihead_attention_kernel.h> // Single_cta, Block_info_padded

namespace fmha
{
namespace ws_sm120
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
struct Compute
{
    using Shared = typename Kernel_traits::Shared;

    using Cbr_q = typename Kernel_traits::Circular_buffer_q_reader;
    using Cbr_k = typename Kernel_traits::Circular_buffer_k_reader;
    using Cbr_v = typename Kernel_traits::Circular_buffer_v_reader;

    using Traits_p = typename Kernel_traits::Traits_p;
    using Traits_o = typename Kernel_traits::Traits_o;
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    using Mma_tile_p = typename Kernel_traits::Mma_tile_p;
    using Mma_tile_o = typename Kernel_traits::Mma_tile_o;
    // MMA tile for the dv-chunked V read (MMAS_K kv-steps x MMAS_N=64/16 dv tiles).
    using Mma_tile_v = typename Kernel_traits::Mma_tile_v;

    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;

    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;

    enum
    {
        STEP_Q = Kernel_traits::STEP_Q
    };

    enum
    {
        STEP_KV = Kernel_traits::STEP_KV
    };

    enum
    {
        CAUSAL_MASK = Kernel_traits::CAUSAL_MASK
    };

    enum
    {
        ENABLE_SKIP_SOFTMAX = Kernel_traits::ENABLE_SKIP_SOFTMAX
    };

    enum
    {
        CHECK_NEG_INF = Kernel_traits::SLIDING_WINDOW_ATTENTION || Kernel_traits::CUSTOM_MASK
    };

    inline __device__ Compute() {}

    // Run on the consumer warps. tidx is the thread index within the
    // consumer group (0 .. NUM_CONSUMER_WARPS*32 - 1).
    template <typename Params>
    inline __device__ void run(int tidx, Shared* shared, Params const& params)
    {
        // Block / head / batch indexing -- same as noloop_tiled.h.
        int const bidb = blockIdx.z;
        int const bidh = blockIdx.y;
        int const q_loop = blockIdx.x; // 1 CTA per (B, H, Q-tile)

        fused_multihead_attention::Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);

        int const q_sequence_start = q_loop * STEP_Q + (binfo.actual_kv_seqlen - binfo.actual_q_seqlen);
        if (binfo.stop_early(q_loop * STEP_Q))
        {
            return;
        }

        // Mask + softmax setup.
        fmha::Mask_dispatcher<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION, /*IS_MTP=*/false> mask(
            params, binfo, tidx);
        // Initialize the mask's query-row offset for this Q-tile. Without this,
        // the causal diagonal defaults to q_sequence_start=0 and every Q-tile
        // after the first masks against the wrong row range. (noloop_tiled.h
        // does the same via mask.load() before the kv loop.)
        mask.load(q_sequence_start);
        // softmax tail buffer is in shared->smem_v's tail; noloop_tiled
        // gives softmax `smem_[Smem_tile_q::BYTES_PER_TILE]` which is the
        // K/V smem region. On skip_softmax we don't share -- softmax does not
        // touch the K/V smem ring. If Softmax::USE_SHARED_MEMORY is needed,
        // we'd allocate a separate softmax_scratch buffer in Shared (TODO).
        Softmax softmax(params, /*smem_scratch=*/nullptr, bidb, tidx);
        static_assert(!Softmax::USE_SHARED_MEMORY,
            "skip_softmax consumer needs Softmax::USE_SHARED_MEMORY = false; if your "
            "kernel_traits enables it, add a softmax_scratch buffer to Shared.");

        // Per-granular-buffer ring readers (Q/K stream the head dim, V streams
        // kv-positions; each cycles GRANULAR_DEPTH buffers).
        Cbr_q cbr_q(&shared->q_barriers);
        Cbr_k cbr_k(&shared->k_barriers);
        Cbr_v cbr_v(&shared->v_barriers);

        // Smem tiles constructed at the granular tile base (buffer 0). Each
        // tile advances its internal read buffer via move_to_next_read_buffer
        // in lockstep with the per-chunk barrier handshake below. The producer
        // (dma_sync_mma.h) TMAs chunk c into buffer (c % GRANULAR_DEPTH).
        Smem_tile_q smem_q(shared->q_buf(0), tidx);
        Smem_tile_k smem_k(shared->k_buf(0), tidx);
        Smem_tile_v smem_v(shared->v_buf(0), tidx);

        // ----- Per-row state shared by the whole KV loop --------------------
        fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
        using Acc_type_o = typename Traits_o::Accumulator_type;
        fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

        fmha::Tile_o_normalizer<Traits_o, Cta_tile_o> acc_o_normalizer(params, binfo);
        float global_max[Softmax::ROWS_PER_THREAD];
        float global_sum[Softmax::ROWS_PER_THREAD];

        // Tail-chunk MMA validity bound for BMM1 (head-dim chunking). When
        // VALID_K is a multiple of the chunk K (the common case, e.g. D=256,
        // chunk=64), all MMAs in every chunk are valid; otherwise the last
        // chunk only runs its first BMM1_TAIL_MMAS_K_BOUND MMAs.
        constexpr int BMM1_VALID_MMAS_K = Mma_tile_p::VALID_MMAS_K;
        constexpr int BMM1_TAIL_MMAS_K_BOUND
            = BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K ? BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K : Mma_tile_p::MMAS_K;

        // Skip-softmax: precompute log(threshold/L) once.
        float const skip_softmax_log_threshold = ENABLE_SKIP_SOFTMAX
            ? __logf(params.skip_softmax_threshold_scale_factor / static_cast<float>(binfo.actual_kv_seqlen))
            : 0.0f;

        int const valid_seqlen
            = CAUSAL_MASK ? min(q_sequence_start + Cta_tile_p::M, binfo.actual_kv_seqlen) : binfo.actual_kv_seqlen;
        int const kv_loop_start = 0;
        int const kv_loop_end = fmha::div_up(valid_seqlen, int(Cta_tile_p::N)) * int(Cta_tile_p::N);
        int const kv_mask_loop_start = int(q_sequence_start / Cta_tile_p::N) * Cta_tile_p::N;

#ifdef SKIP_SOFTMAX_STAT
        // Skip-softmax block counters (compiled only in a -DSKIP_SOFTMAX_STAT
        // build). tile_negligible is per-warp and uniform within a warp, so
        // every thread keeps an identical local tally; only the elected thread
        // (tidx == 0) flushes to global below, reporting the elected consumer
        // warp's rate as the CTA proxy (same convention as noloop_tiled.h).
        [[maybe_unused]] uint32_t skip_softmax_total = 0;
        [[maybe_unused]] uint32_t skip_softmax_skipped = 0;
#endif

        // ----- KV loop ------------------------------------------------------
        for (int kv_loop = kv_loop_start; kv_loop < kv_loop_end; kv_loop += Cta_tile_p::N)
        {
            bool const first_step = (kv_loop == kv_loop_start);
            bool tile_negligible = false;
            bool const apply_mask = params.has_alibi || (kv_loop >= kv_mask_loop_start);

            fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            using Acc_type_p = typename Traits_p::Accumulator_type;
            fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

            mask.move_to_offset(kv_loop);

            typename Smem_tile_q::Fragment frag_q[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
            typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];

            // ---- BMM1: stream the head dim in NUM_BMM1_CHUNKS granular chunks.
            // Each chunk c is a separate TMA-filled buffer (chunk % GRANULAR_DEPTH);
            // we wait on its produced barrier, do MMAS_K MMAs, then signal
            // consumed (all consumer threads arrive -> doubles as the
            // pre-recycle sync) and advance to the next granular read buffer.
            // This replaces noloop_tiled.h's per-chunk LDGSTS reload +
            // ldgdepbar + __syncthreads.
#pragma unroll
            for (int chunk = 0; chunk < Kernel_traits::NUM_BMM1_CHUNKS; ++chunk)
            {
                bool const is_tail = (chunk == Kernel_traits::NUM_BMM1_CHUNKS - 1);
                int const k_slot = cbr_k.wait();
                int const q_slot = cbr_q.wait();
#pragma unroll
                for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
                {
                    smem_q.load(frag_q[ki], ki);
                    smem_k.load(frag_k[ki], ki);
                    if (!is_tail || Cta_tile_p::VALID_K % Cta_tile_p::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND)
                    {
                        fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
                    }
                }
                // Every consumer thread arrives on the consumed barrier (count
                // == CONSUMER_THREADS), so the producer cannot overwrite the
                // buffer until all reads are done.
                cbr_k.complete(/*arrive=*/1, k_slot);
                cbr_k.advance();
                cbr_q.complete(/*arrive=*/1, q_slot);
                cbr_q.advance();
                smem_k.move_to_next_read_buffer();
                smem_q.move_to_next_read_buffer();
            }

            // ---- Softmax ----
            softmax.unpack(acc_p);
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

            // Hoist frag_p (lifted from noloop_tiled.h; skip-softmax pack lives in tail gate).
            fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Kernel_traits::TOTAL_BMM2_MMAS_K][Mma_tile_o::MMAS_M];

            if (first_step)
            {
                softmax.template reduce<fmha::Max_>(global_max);
                softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
                softmax.template reduce<fmha::Sum_>(global_sum);
                if constexpr (ENABLE_SKIP_SOFTMAX)
                {
                    softmax.pack(frag_p);
                }
            }
            else
            {
                float tmp[Softmax::ROWS_PER_THREAD];
#pragma unroll
                for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                {
                    tmp[i] = global_max[i];
                }
                softmax.template reduce<fmha::Max_>(global_max);

                // Per-warp skip-softmax vote.
                if constexpr (ENABLE_SKIP_SOFTMAX)
                {
#ifdef SKIP_SOFTMAX_STAT
                    ++skip_softmax_total;
#endif
                    bool skip = ((global_max[0] - tmp[0]) < skip_softmax_log_threshold);
#pragma unroll
                    for (int i = 1; i < Softmax::ROWS_PER_THREAD; i++)
                    {
                        skip = skip & ((global_max[i] - tmp[i]) < skip_softmax_log_threshold);
                    }
                    tile_negligible = __all_sync(0xffffffffu, skip);
                    if (tile_negligible)
                    {
#ifdef SKIP_SOFTMAX_STAT
                        ++skip_softmax_skipped;
#endif
#pragma unroll
                        for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                        {
                            global_max[i] = tmp[i];
                        }
                    }
                }

                if (!tile_negligible)
                {
                    acc_o_normalizer.update(acc_o, global_max, tmp, global_sum);
                    softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
#pragma unroll
                    for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                    {
                        tmp[i] = global_sum[i];
                        global_sum[i] = 0.f;
                    }
                    softmax.template reduce<fmha::Sum_>(global_sum);
#pragma unroll
                    for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++)
                    {
                        global_sum[i] += tmp[i];
                    }
                    if constexpr (ENABLE_SKIP_SOFTMAX)
                    {
                        softmax.pack(frag_p);
                    }
                }
            }

            // Baseline (non-skip-softmax) pack: same location as noloop_tiled.h.
            if constexpr (!ENABLE_SKIP_SOFTMAX)
            {
                softmax.pack(frag_p);
            }

            // ---- BMM2: V is tiled in DV (outer) x kv-positions (inner).
            // Each sub-tile is a [kv-chunk, dv-chunk=64] 128-byte-row granular
            // buffer. For dv-chunk dvc we contract all kv (across the kv-chunks)
            // into the acc_o columns [dvc*MMAS_N .. +MMAS_N). frag_p (the
            // softmax probs over all kv) is indexed by the global k-step
            // kvc*MMAS_K + ki. We ALWAYS consume the V sub-tiles (to drain the
            // producer pipeline) but skip the HMMAs on a negligible tile.
            typename Smem_tile_v::Fragment frag_v[Mma_tile_v::MMAS_K][Mma_tile_v::VALID_MMAS_N];

            bool const do_bmm2 = !(ENABLE_SKIP_SOFTMAX && tile_negligible);
#pragma unroll
            for (int dvc = 0; dvc < Kernel_traits::NUM_BMM2_DV_CHUNKS; ++dvc)
            {
#pragma unroll
                for (int kvc = 0; kvc < Kernel_traits::NUM_BMM2_KV_CHUNKS; ++kvc)
                {
                    int const v_slot = cbr_v.wait();
                    if (do_bmm2)
                    {
#pragma unroll
                        for (int ki = 0; ki < Mma_tile_v::MMAS_K; ++ki)
                        {
                            int const p_ki = kvc * Mma_tile_v::MMAS_K + ki; // global frag_p k-step
                            smem_v.load(frag_v[ki], ki);
#pragma unroll
                            for (int ni = 0; ni < Mma_tile_v::VALID_MMAS_N; ++ni)
                            {
                                int const acc_ni = dvc * Mma_tile_v::VALID_MMAS_N + ni;
#pragma unroll
                                for (int mi = 0; mi < Mma_tile_o::MMAS_M; ++mi)
                                {
                                    acc_o[mi][acc_ni].mma(frag_p[p_ki][mi], frag_v[ki][ni]);
                                }
                            }
                        }
                    }
                    cbr_v.complete(/*arrive=*/1, v_slot);
                    cbr_v.advance();
                    smem_v.move_to_next_read_buffer();
                }
            }
        }

#ifdef SKIP_SOFTMAX_STAT
        // Flush this CTA's skip tally to the global counters. Only the elected
        // thread (tidx == 0 of the consumer group) writes, so we record the
        // first consumer warp's tally as the CTA proxy (noloop_tiled.h does the
        // same). trtllm.py reads + prints skipped/total per layer when
        // TRTLLM_PRINT_SKIP_SOFTMAX_STAT=1.
        if constexpr (ENABLE_SKIP_SOFTMAX)
        {
            if (tidx == 0 && params.skip_softmax_total_blocks != nullptr)
            {
                atomicAdd(params.skip_softmax_total_blocks, skip_softmax_total);
                atomicAdd(params.skip_softmax_skipped_blocks, skip_softmax_skipped);
            }
        }
#endif

        // ---- Epilogue: normalize acc_o by global_sum, store O ----
        // Ported from noloop_tiled.h, with two skip_softmax adaptations:
        //   * __syncthreads() -> named_barrier over the CONSUMER_THREADS group
        //     only (the producer warp is not part of the epilogue and must not
        //     be caught in a CTA-wide barrier).
        //   * Smem_tile_o aliases the start of the smem region (q/k/v buffers
        //     are free once the kv-loop is done); it needs
        //     Smem_tile_o::BYTES_PER_TILE bytes, which fits within the
        //     q+k+v span ahead of the barrier arrays.
        acc_o_normalizer.update_sum(global_max, global_sum);
        acc_o_normalizer.final_update(acc_o, global_sum);

        Gmem_tile_o gmem_o(params, binfo, tidx, q_loop * Gmem_tile_o::ROWS);
        Smem_tile_o smem_o(&shared->smem_q[0], tidx);

#pragma unroll
        for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
        {
            // Swizzle the elements and do the final cross-warp reduction.
            smem_o.store(acc_o, ii);
            // Make sure the data is in shared memory (consumer group only).
            fmha::named_barrier_wait(Kernel_traits::CONSUMER_SYNC_BARRIER_ID, Kernel_traits::CONSUMER_THREADS);

            uint4 out[Gmem_tile_o::STGS_PER_LOOP];
            smem_o.load(out);

            // Make sure the data was read from shared memory before reuse.
            if (ii < Gmem_tile_o::LOOPS - 1)
            {
                fmha::named_barrier_wait(Kernel_traits::CONSUMER_SYNC_BARRIER_ID, Kernel_traits::CONSUMER_THREADS);
            }

            gmem_o.store(out, ii);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha
