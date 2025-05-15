/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fused_multihead_attention_kernel.h>

#include "gmem_tile_d.h"
#include "smem_tile_d.h"

#include "philox.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// Strategy 0 is more efficient at batches with short sequences, as work is distributed more evenly.
// Strategy 1 is faster if all sequences are full
template <int CHUNKS, typename Cta_tile, int Strategy>
struct Noloop_traits
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is the more efficient way of breaking up the problem for small batches,
//   which is more relevant for the benchmark.
// But the additional overhead reduces performance a bit for (close to) full batches.
template <int CHUNKS, typename Cta_tile>
struct Noloop_traits<CHUNKS, Cta_tile, 0>
{
    // Interpretation of Cta_tile dims, i.e. Cta_tile_p:
    enum
    {
        STEP = Cta_tile::M
    };

    enum
    {
        SEQLEN = Cta_tile::N
    };

    template <typename Block_info>
    inline __device__ Noloop_traits(int const bidc, Block_info const& binfo)
        : bidc_(bidc)
    {
        int const seqlen = binfo.actual_seqlen;
        int const steps = (seqlen + STEP - 1) / STEP;
        int const steps_per_chunk = (steps + CHUNKS - 1) / CHUNKS;

        int const step_begin = bidc_ * steps_per_chunk;
        int const step_end = min(steps, (bidc_ + 1) * steps_per_chunk);
        int const actual_steps = max(0, step_end - step_begin);
        loop_offset_ = step_begin;
        num_steps_ = actual_steps;
    }

    template <typename... Tiles>
    inline __device__ void move_all(Tiles&... tiles) const
    {
        using expand_type = int[];
        for (int s = 0; s < loop_offset_; s++)
        {
            expand_type{(tiles.move(), 0)...};
        }
    }

    inline __device__ int get_idx_dk() const
    {
        // return bidc_;
        return bidc_ * 2 + 0;
    }

    inline __device__ int get_idx_dv() const
    {
        // return CHUNKS + bidc_;
        return bidc_ * 2 + 1;
    }

    inline __device__ int offset_loop_count(int const l)
    {
        // convert loop counter to position in the outer sequence
        return (loop_offset_ + l) * STEP;
    }

    uint32_t const bidc_;
    int loop_offset_;
    int num_steps_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Noloop_traits<2, Cta_tile, 1>
{
    // Interpretation of Cta_tile dims, i.e. Cta_tile_p:
    enum
    {
        STEP = Cta_tile::M
    };

    enum
    {
        SEQLEN = Cta_tile::N
    };

    enum
    {
        CHUNKS = 2
    };

    // The size of the subsequence this CTA is processing
    enum
    {
        SUBSEQ = SEQLEN / CHUNKS
    };

    static_assert(SUBSEQ * CHUNKS == SEQLEN);

    // The number of steps to process the subsequence
    enum
    {
        NUM_STEPS = SUBSEQ / STEP
    };

    static_assert(NUM_STEPS * Cta_tile::M == SUBSEQ);

    template <typename Block_info>
    inline __device__ Noloop_traits(int const bidc, Block_info const&)
        : loop_offset_(NUM_STEPS * bidc)
        , bidc_(bidc)
    {
    }

    template <typename... Tiles>
    inline __device__ void move_all(Tiles&... tiles) const
    {
        using expand_type = int[];
        for (int s = 0; s < loop_offset_; s++)
        {
            expand_type{(tiles.move(), 0)...};
        }
    }

    inline __device__ int get_idx_dk() const
    {
        // return bidc_;
        return bidc_ * 2 + 0;
    }

    inline __device__ int get_idx_dv() const
    {
        // return CHUNKS + bidc_;
        return bidc_ * 2 + 1;
    }

    inline __device__ int offset_loop_count(int const l)
    {
        // convert loop counter to position in the outer sequence
        return (loop_offset_ + l) * STEP;
    }

    int const loop_offset_;
    uint32_t const bidc_;
    int const num_steps_ = NUM_STEPS;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Noloop_traits<3, Cta_tile, 1>
{
    // Interpretation of Cta_tile dims, i.e. Cta_tile_p:
    enum
    {
        STEP = Cta_tile::M
    };

    enum
    {
        SEQLEN = Cta_tile::N
    };

    enum
    {
        CHUNKS = 3
    };

    static_assert(STEP == 16 && SEQLEN == 512);

    // 3 chunks: 512 = 160 + 160 + 192

    template <typename Block_info>
    inline __device__ Noloop_traits(int const bidc, Block_info const&)
        : bidc_(bidc)
        , num_steps_(bidc < 2 ? 11 : 10)
        , loop_offset_(bidc * 11)
    {
    }

    template <typename... Tiles>
    inline __device__ void move_all(Tiles&... tiles) const
    {
        using expand_type = int[];
        for (int s = 0; s < loop_offset_; s++)
        {
            expand_type{(tiles.move(), 0)...};
        }
    }

    inline __device__ int get_idx_dk() const
    {
        // return bidc_;
        return bidc_ * 2 + 0;
    }

    inline __device__ int get_idx_dv() const
    {
        // return CHUNKS + bidc_;
        return bidc_ * 2 + 1;
    }

    inline __device__ int offset_loop_count(int const l)
    {
        // convert loop counter to position in the outer sequence
        return (loop_offset_ + l) * STEP;
    }

    int const loop_offset_;
    uint32_t const bidc_;
    int const num_steps_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
std::tuple<int, int, int, int, int, int> work_dist(int const S, int const total_ctas, int const heads_total)
{

    int const STEPS_PER_HEAD = (S + Kernel_traits::Cta_tile_p::M - 1) / Kernel_traits::Cta_tile_p::M;

    int const num_full_heads = heads_total / total_ctas;
    int const heads_last_wave = heads_total % total_ctas; // [0, total_ctas)

    int num_main_groups = 0;
    int main_steps = 0;
    int rest_steps = 0;
    if (heads_last_wave > 0)
    {
        // Number of CTA groups that process within heads.
        num_main_groups = total_ctas / heads_last_wave;
        // Remaining CTAs that process between heads.
        int const rest_ctas = total_ctas - (heads_last_wave * num_main_groups);
        if (rest_ctas == 0)
        {
            // We have exactly "num_main_groups" CTAs to process each of the remaining heads.

            main_steps = (STEPS_PER_HEAD + num_main_groups - 1) / num_main_groups;
            num_main_groups = STEPS_PER_HEAD / main_steps; // main_step > 0!
            rest_steps = STEPS_PER_HEAD % main_steps;
        }
        else
        {
            // Ideal number of steps if we could load-balance as evenly as possible.
            int const steps_ideal = (heads_last_wave * STEPS_PER_HEAD + total_ctas - 1) / total_ctas;
            // Iterations that a "rest" CTA has to do at most.
            int const max_rest_iters = (heads_last_wave + rest_ctas - 1) / rest_ctas;
            // Find the first step distribution, s.t. the maximum work of
            //  the "rest" CTAs is less than the work of the main CTAs.
            main_steps = steps_ideal;
            rest_steps = STEPS_PER_HEAD - main_steps * num_main_groups;
            for (; main_steps * num_main_groups < STEPS_PER_HEAD; main_steps++)
            {
                rest_steps = STEPS_PER_HEAD - main_steps * num_main_groups;
                int const max_rest_total_steps = rest_steps * max_rest_iters;
                if (max_rest_total_steps < main_steps)
                    break;
            }
            rest_steps = STEPS_PER_HEAD - main_steps * num_main_groups;
        }
    }

    using Traits = typename Kernel_traits::Traits_p;
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;

    int const max_steps = STEPS_PER_HEAD * num_full_heads + std::max(main_steps, rest_steps);
    int const elts_per_thread_per_step = Mma_tile_p::MMAS_M * Mma_tile_p::MMAS_N * 8;
    int const elts_per_thread = max_steps * elts_per_thread_per_step;

    return {num_full_heads, num_main_groups, heads_last_wave, main_steps, rest_steps, elts_per_thread};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
