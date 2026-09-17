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

// Kernel_traits for the skip_softmax sm_120 / sm_121 warp-specialized FMHA.
//
// Wraps the existing fmha::Kernel_traits_ template (which already provides
// LDGSTS-friendly Smem_tile_a/b/v with ldmatrix swizzle and the right
// Cta_tile / Mma_tile / fragment shapes) and layers on the warp-spec
// pieces:
//   * Shared struct with smem tiles + entry-produced / entry-consumed
//     mbarrier arrays.
//   * Circular_buffer_{q,k,v}_{reader,writer} type aliases against the
//     existing fmha::ws::CircularBuffer infrastructure.
//   * Named-barrier ids (collision-safe with the existing skip-softmax
//     barrier ids 0x3 / 0x4 used on the non-warpspec tiled kernel).
//
// What this header does NOT include:
//   * Host-side TMA descriptor setup (phase 3).
//   * Persistence of which slot has which kv_loop offset (handled by
//     ring writer state, not traits).

#include <fmha/kernel_traits.h>
#include <fmha/utils.h>
#include <fmha/warpspec/circular_buffer.h>

namespace fmha
{
namespace ws_sm120
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits (e.g. Ampere_hmma_bf16_traits) -- shared with the
    // non-warpspec tiled kernel for sm_120.
    typename Traits_,
    // Sequence length upper bound (e.g. 8192 for typical chunked prefill).
    int S,
    // Hidden head dim (e.g. 128, 192, 256).
    int VALID_D_,
    // Hidden head dim of V (= D in standard MHA).
    int VALID_DV_,
    // The iteration step of the outer Q loop.
    int STEP_Q_,
    // The number of vertical warps in the compute group (consumer-side).
    int WARPS_M_,
    // The number of horizontal warps in the compute group.
    int WARPS_N_,
    // The version of the kernel (passes through to Kernel_traits_).
    int VERSION_,
    // The mask version of the kernel.
    int MASK_VERSION_,
    // Skip-softmax knob.
    bool ENABLE_SKIP_SOFTMAX_ = false,
    // Producer warp count -- single 32-thread warp by default.
    int NUM_PRODUCER_WARPS_ = 1>
struct Kernel_traits_skip_softmax_sm120
{

    // Compose the existing PACKED_QKV tiled kernel traits with our skip_softmax
    // overrides. Kernel_traits_v2 (in fmha/kernel_traits.h) bakes in
    // fmha::v2::Gmem_tile_qkv for Q / K / V, which is the right gmem tile
    // for the PACKED_QKV input layout TRT-LLM passes for Qwen3.5 prefill.
    //
    // FLAGS: bit 0x1   = USE_LDGSTS_Q
    //        bit 0x2   = USE_LDGSTS_K
    //        bit 0x4   = USE_LDGSTS_V
    //                    Skip_softmax leaves all three LDGSTS bits OFF; the
    //                    producer warp issues TMA, not LDGSTS. The smem
    //                    layout is governed by BYTES_PER_LDG (=16,
    //                    independent of USE_LDGSTS) and the buffer count
    //                    (controlled by USE_GRANULAR_TILING), so turning
    //                    LDGSTS off does NOT change the layout the
    //                    consumer reads via ldmatrix. It also satisfies
    //                    gmem_tile_qkv_packed.h's static assertion that
    //                    USE_LDGSTS=>(PRED_REGS==1 || IS_HOPPER), which
    //                    we'd fail on sm_120 with non-trivial Q tiles.
    //        bit 0x200 = NO_LOOP (we are a no-loop kernel)
    //        bit 0x1000= USE_GRANULAR_TILING (matches noloop_tiled.h)
    static constexpr uint32_t TILED_FLAGS = 0x200u // NO_LOOP
        | 0x1000u                                  // USE_GRANULAR_TILING
        ;

    using Base = fmha::Kernel_traits_v2<Traits_,
        /*S=*/S, /*D=*/VALID_D_, /*DV=*/VALID_DV_,
        /*STEP=*/STEP_Q_,
        /*WARPS_M=*/WARPS_M_, /*WARPS_N=*/WARPS_N_,
        /*CTAS_PER_HEAD=*/1,
        /*FLAGS=*/TILED_FLAGS,
        /*MASK_VERSION=*/MASK_VERSION_,
        /*BMM2_FP16_EPILOGUE=*/true,
        /*OutputType=*/typename Traits_::A_type,
        /*SAGE_BLOCK_SIZE_Q=*/0,
        /*SAGE_BLOCK_SIZE_K=*/0,
        /*SAGE_BLOCK_SIZE_V=*/0,
        /*ENABLE_SKIP_SOFTMAX=*/ENABLE_SKIP_SOFTMAX_>;

    // Carry through the math types -- these are what compute_sync_mma.h needs.
    using Traits_p = typename Base::Traits_p;
    using Traits_o = typename Base::Traits_o;
    using Traits_e = typename Base::Traits_e;
    using Cta_tile_p = typename Base::Cta_tile_p;
    using Cta_tile_o = typename Base::Cta_tile_o;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using Mma_tile_o = typename Base::Mma_tile_o;

    // Smem tiles: reuse the existing tiled-kernel types and their buffer counts
    // (Base::BUFFERS_PER_TILE_SMEM_*); the swizzle and the consumer's ldmatrix
    // access patterns are independent of the buffer count.
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_o = typename Base::Smem_tile_o;

    // ----- V re-tiled to 64-wide DV chunks (6c.3) -----------------------------
    //
    // The Base Smem_tile_v packs the full DV (256) into the lead dim, giving
    // 512-byte smem rows that no TMA swizzle mode can reproduce (the encode caps
    // the leading box dim at the 128-byte swizzle width). We instead tile DV
    // into BMM2_DV_CHUNK=64-wide groups so the V smem tile has LEAD_DIM=64 ->
    // 128-byte rows == the same proven layout as K (TMA-128B-swizzle fillable),
    // and the existing Smem_tile_v `N==64` ldsmt read path applies unchanged.
    static constexpr int BMM2_DV_CHUNK = 64;
    using Cta_tile_v = typename Traits_o::template Cta_tile_extd<Cta_tile_o::M, BMM2_DV_CHUNK, Cta_tile_o::K,
        BMM2_DV_CHUNK, S, WARPS_M_, 1, WARPS_N_>;
    using Smem_tile_v = fmha::Smem_tile_v<Traits_o, Cta_tile_v, Base::BUFFERS_PER_TILE_SMEM_V>;
    // MMA tile for the dv-chunk V read (MMAS_K kv-steps, MMAS_N = 64/16 dv tiles).
    using Mma_tile_v = typename Traits_o::template Mma_tile<Cta_tile_v>;

    using Gmem_tile_o = typename Base::Gmem_tile_o;

    enum
    {
        VALID_D = Base::VALID_D
    };

    enum
    {
        D = Base::D
    };

    enum
    {
        VALID_DV = Base::VALID_DV
    };

    enum
    {
        DV = Base::DV
    };

    enum
    {
        STEP_Q = STEP_Q_
    };

    enum
    {
        STEP_KV = Cta_tile_p::N
    };

    enum
    {
        VERSION = VERSION_
    };

    enum
    {
        MASK_VERSION = MASK_VERSION_
    };

    enum
    {
        CAUSAL_MASK = Base::CAUSAL_MASK
    };

    enum
    {
        SLIDING_WINDOW_ATTENTION = Base::SLIDING_WINDOW_ATTENTION
    };

    enum
    {
        BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION = Base::BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION
    };

    enum
    {
        CUSTOM_MASK = Base::CUSTOM_MASK
    };

    enum
    {
        ELEMENT_BYTES = sizeof(typename Traits_p::A_type)
    };

    enum
    {
        TOTAL_BMM2_MMAS_K = Base::TOTAL_BMM2_MMAS_K
    };

    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Base::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    enum
    {
        IS_MTP = Base::IS_MTP
    };

    // Skip-softmax knob.
    static constexpr bool ENABLE_SKIP_SOFTMAX = ENABLE_SKIP_SOFTMAX_;

    // Producer + consumer warp layout.
    enum
    {
        NUM_PRODUCER_WARPS = NUM_PRODUCER_WARPS_
    };

    enum
    {
        NUM_CONSUMER_WARPS = WARPS_M_ * WARPS_N_
    };

    enum
    {
        THREADS = (NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS) * 32
    };

    // Named-barrier ids. Collision-safe with the existing skip-softmax
    // barriers (0x3, 0x4 on the non-warpspec path) since we don't run both
    // kernels in the same launch.
    static constexpr int DMA_SYNC_BARRIER_ID = 0x1;
    static constexpr int MMA_SYNC_BARRIER_ID = 0x2;

    // Named-barrier id for the consumer group's pre-recycle sync (all consumer
    // warps must finish reading a granular buffer before tid0 lets the producer
    // overwrite it).
    static constexpr int CONSUMER_SYNC_BARRIER_ID = 0x5;

    // Single CTA cluster (no DSMEM on consumer Blackwell -- CTAS_PER_CGA=1).
    static constexpr int CTAS_PER_CGA = 1;

    enum
    {
        CONSUMER_THREADS = NUM_CONSUMER_WARPS * 32
    };

    // ----- Granular head-dim / kv-position chunking ---------------------------
    //
    // The consumer reuses the existing LDGSTS Smem_tile_q/k/v which, with
    // USE_GRANULAR_TILING, stream the contraction dim in chunks through a
    // BUFFERS_PER_TILE-deep ping-pong (== GRANULAR_DEPTH below). The skip_softmax
    // TMA producer fills those exact granular buffers chunk by chunk -- it does
    // NOT add a ring on top. So the barrier depth == the granular buffer count.
    static constexpr int GRANULAR_DEPTH = Smem_tile_k::BUFFERS_PER_TILE;
    static_assert(GRANULAR_DEPTH == Smem_tile_q::BUFFERS_PER_TILE && GRANULAR_DEPTH == Smem_tile_v::BUFFERS_PER_TILE,
        "skip_softmax assumes Q/K/V share the same granular buffer depth.");

    // Bytes of one granular buffer (one chunk) for Q / K / V.
    static constexpr int BYTES_PER_BUFFER_Q = Smem_tile_q::BYTES_PER_BUFFER;
    static constexpr int BYTES_PER_BUFFER_K = Smem_tile_k::BYTES_PER_BUFFER;
    static constexpr int BYTES_PER_BUFFER_V = Smem_tile_v::BYTES_PER_BUFFER;

    // BMM1 streams the head dim in chunks of Cta_tile_p::K (=64) elements.
    static constexpr int BMM1_CHUNK_ELTS = Cta_tile_p::K; // head-dim elements / chunk
    static constexpr int NUM_BMM1_CHUNKS = Mma_tile_p::VALID_MMAS_K / Mma_tile_p::MMAS_K;

    // BMM2 V is tiled in BOTH kv-positions (Cta_tile_o::K = 32 each) and DV
    // (BMM2_DV_CHUNK = 64 each). Each V sub-tile is one [kv-chunk, dv-chunk]
    // granular buffer ([32, 64] -> 128-byte rows). The producer streams
    // NUM_BMM2_DV_CHUNKS * NUM_BMM2_KV_CHUNKS sub-tiles per kv-tile.
    static constexpr int BMM2_KV_CHUNK_ELTS = Cta_tile_o::K;                          // 32
    static constexpr int NUM_BMM2_KV_CHUNKS = TOTAL_BMM2_MMAS_K / Mma_tile_o::MMAS_K; // 4
    static constexpr int NUM_BMM2_DV_CHUNKS = VALID_DV / BMM2_DV_CHUNK;               // 4
    static constexpr int NUM_BMM2_CHUNKS = NUM_BMM2_KV_CHUNKS * NUM_BMM2_DV_CHUNKS;   // 16

    using Circular_buffer_q_reader = typename fmha::ws::CircularBuffer<GRANULAR_DEPTH, CTAS_PER_CGA>::Reader;
    using Circular_buffer_q_writer = typename fmha::ws::CircularBuffer<GRANULAR_DEPTH, CTAS_PER_CGA>::Writer;
    using Circular_buffer_k_reader = typename fmha::ws::CircularBuffer<GRANULAR_DEPTH, CTAS_PER_CGA>::Reader;
    using Circular_buffer_k_writer = typename fmha::ws::CircularBuffer<GRANULAR_DEPTH, CTAS_PER_CGA>::Writer;
    using Circular_buffer_v_reader = typename fmha::ws::CircularBuffer<GRANULAR_DEPTH, CTAS_PER_CGA>::Reader;
    using Circular_buffer_v_writer = typename fmha::ws::CircularBuffer<GRANULAR_DEPTH, CTAS_PER_CGA>::Writer;

    // Shared struct: the granular smem tiles (flat, == GRANULAR_DEPTH buffers
    // laid out contiguously) + barrier arrays. The TMA producer writes chunk c
    // into buffer (c % GRANULAR_DEPTH); the consumer's Smem_tile cycles its
    // internal read buffer via move_to_next_read_buffer in lockstep.
    struct __align__(128) Shared
    {
        uint8_t smem_q[Smem_tile_q::BYTES_PER_TILE];
        uint8_t smem_k[Smem_tile_k::BYTES_PER_TILE];
        uint8_t smem_v[Smem_tile_v::BYTES_PER_TILE];

        fmha::ws::CircularBufferBarriers<GRANULAR_DEPTH> q_barriers;
        fmha::ws::CircularBufferBarriers<GRANULAR_DEPTH> k_barriers;
        fmha::ws::CircularBufferBarriers<GRANULAR_DEPTH> v_barriers;

        // Smem address of granular buffer `slot` for each tensor (producer TMA
        // destination / consumer tile base + slot stride).
        inline __device__ uint8_t* q_buf(int slot)
        {
            return &smem_q[slot * BYTES_PER_BUFFER_Q];
        }
        inline __device__ uint8_t* k_buf(int slot)
        {
            return &smem_k[slot * BYTES_PER_BUFFER_K];
        }
        inline __device__ uint8_t* v_buf(int slot)
        {
            return &smem_v[slot * BYTES_PER_BUFFER_V];
        }

        // Initialize all mbarriers. Called by thread 0 of the CTA at startup.
        //   entryProducedBarriers: count 1 -- the elect-one DMA thread arms the
        //     transaction count via tmaReserve; the TMA completion's tx-bytes
        //     arrival flips the barrier.
        //   entryConsumedBarriers: count CONSUMER_THREADS -- every consumer
        //     thread arrives once it has finished reading the buffer, which also
        //     serves as the pre-recycle sync (no separate __syncthreads needed).
        inline __device__ void init(bool tid0)
        {
            if (tid0)
            {
#pragma unroll
                for (int i = 0; i < GRANULAR_DEPTH; i++)
                {
                    fmha::bar_create(&q_barriers.entryProducedBarriers[i], 1);
                    fmha::bar_create(&q_barriers.entryConsumedBarriers[i], CONSUMER_THREADS);
                    fmha::bar_create(&k_barriers.entryProducedBarriers[i], 1);
                    fmha::bar_create(&k_barriers.entryConsumedBarriers[i], CONSUMER_THREADS);
                    fmha::bar_create(&v_barriers.entryProducedBarriers[i], 1);
                    fmha::bar_create(&v_barriers.entryConsumedBarriers[i], CONSUMER_THREADS);
                }
            }
        }
    };

    // Pad to align. The non-Hopper kernel allocates BYTES_PER_SMEM in the
    // extern __shared__ block; the skip_softmax version uses sizeof(Shared).
    enum
    {
        BYTES_PER_SMEM = sizeof(Shared)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws_sm120
} // namespace fmha
