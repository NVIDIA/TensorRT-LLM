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
#include <fmha/gmem_tile_qkv_packed.h>
#include <fmha/hopper/compute_tile.h>
#include <fmha/hopper/gmem_tile_o_packed.h>
#include <fmha/hopper/gmem_tile_qkv_packed.h>
#include <fmha/hopper/smem_tile.h>
#include <fmha/hopper/smem_tile_o.h>
#include <fmha/smem_tile.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // Instruction traits.
    typename Traits_p_,
    // Instruction traits.
    typename Traits_o_,
    // The ldgsts global memory tile for Q, K and V.
    template <typename, typename, int, int, int, int, bool, bool, int, bool> class Gmem_tile_qkv_,
    // The tma global memory tile for Q, K and V.
    template <typename, typename, int, int, int, bool, bool, int> class Gmem_tile_tma_qkv_,
    // The global memory tile for the output.
    template <typename, typename, int> class Gmem_tile_o_,
    // Sequence length.
    int S,
    // The hidden dimension.
    int D,
    // The iteration step of the outer loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The version of the kernel.
    int VERSION_,
    // The mask version of the kernel, (2 denotes dense mask, 3 denotes causal mask)
    int MASK_VERSION_ = 2,
    // The flags to control the behaviour of LDGs.
    uint32_t FLAGS = 0x8u>
struct FMHA_kernel_traits_hopper
{

    // The instruction traits for the Q*K product.
    using Traits_p = Traits_p_;

    // is Q operand in RF for GMMA?
    static constexpr bool GMMA_Q_RF = Traits_p::GMMA_A_RF;

    // is K operand in RF for GMMA?
    static constexpr bool GMMA_K_RF = Traits_p::GMMA_B_RF;

    // The instruction traits for P*V product.
    using Traits_o = Traits_o_;

    // is S operand in RF for GMMA?
    static constexpr bool GMMA_S_RF = Traits_o::GMMA_A_RF;

    // is V operand in RF for GMMA?
    static constexpr bool GMMA_V_RF = Traits_o::GMMA_B_RF;

    // The number of warpgroups along M dimension
    enum
    {
        WARP_GROUP_M = WARPS_M / 4
    };

    // The number of warpgroups along N dimension
    enum
    {
        WARP_GROUP_N = WARPS_N
    };

    // The number of warpgroups along K dimension
    enum
    {
        WARP_GROUP_K = 1
    };

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = typename Traits_p::template Cta_tile<STEP, S, D, WARP_GROUP_M, WARP_GROUP_N, 1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = typename Traits_o::template Cta_tile<STEP, D, S, WARP_GROUP_M, 1, WARP_GROUP_N>;

    // The version.
    enum
    {
        VERSION = VERSION_
    };

    enum
    {
        MASK_VERSION = MASK_VERSION_
    };

    // Whether use causal mask or not.
    enum
    {
        CAUSAL_MASK = MASK_VERSION_ >= 3
    };

    // Whether use the sliding window attention mask or not.
    enum
    {
        SLIDING_WINDOW_ATTENTION = MASK_VERSION_ == 4
    };

    // Do we use LDGSTS for Q, K or V. If not, TMA is used!
    enum
    {
        USE_LDGSTS_Q = (FLAGS & 0x1u) != 0u
    };

    enum
    {
        USE_LDGSTS_K = (FLAGS & 0x2u) != 0u
    };

    enum
    {
        USE_LDGSTS_V = (FLAGS & 0x4u) != 0u
    };

    enum
    {
        USE_TMA_Q = !USE_LDGSTS_Q
    };

    enum
    {
        USE_TMA_K = !USE_LDGSTS_K
    };

    enum
    {
        USE_TMA_V = !USE_LDGSTS_V
    };

    // Do we use one buffer for K and V.
    enum
    {
        SHARE_SMEM_FOR_K_AND_V = 0
    };

    // Do we use the scale max trick.
    enum
    {
        USE_SCALE_MAX = 0
    };

    // Are heads in QKV interleaved, i.e. total x h x 3 x d or total x 3 x h x d.
    enum
    {
        HEADS_INTERLEAVED = (FLAGS & 0x20u) == 0u
    };

    // Use BMM1 softcapping scale or not.
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = (FLAGS & 0x800) != 0u
    };

    // Number of matrix for gmem_tile_qkv
    enum
    {
        NUM_QKV_MATS = 3
    };

    // The global memory tile to load Q.
    // Hopefully we don't need to specialize for Hopper.
    using Gmem_tile_ldgsts_q = Gmem_tile_qkv_<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP, D, D, true,
        HEADS_INTERLEAVED, NUM_QKV_MATS, SLIDING_WINDOW_ATTENTION>;

    // The global memory tile to load Q with TMA.
    using Gmem_tile_tma_q = Gmem_tile_tma_qkv_<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP, D, false,
        HEADS_INTERLEAVED, NUM_QKV_MATS>;

    // Do we use ldgsts gmem tile or tma gmem tile?
    using Gmem_tile_q = typename std::conditional_t<USE_LDGSTS_Q, Gmem_tile_ldgsts_q, Gmem_tile_tma_q>;

    // 2 buffers for Q
    enum
    {
        BUFFERS_PER_SMEM_TILE_Q = 2
    };

    // Q is row major
    using Q_layout = fmha::Row;

    // We know Q is row-major. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_Q
        = Cta_tile_p::K * sizeof(typename Traits_p::A_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    // The shared memory tile to swizzle Q.
    using Smem_tile_ldgsts_q = fmha::Smem_tile_hopper_a<Traits_p, Cta_tile_p, Q_layout, 16, BUFFERS_PER_SMEM_TILE_Q,
        GMMA_DESC_MODE_Q, USE_TMA_Q, GMMA_Q_RF>;

    // The shared memory tile to swizzle Q. TODO: need to update to XMMA.
    using Smem_tile_tma_q = fmha::wip::Smem_tile_hopper_a<Traits_p, Cta_tile_p, Q_layout, BUFFERS_PER_SMEM_TILE_Q,
        GMMA_DESC_MODE_Q, GMMA_Q_RF, USE_LDGSTS_Q>;

    using Smem_tile_q = typename std::conditional_t<USE_LDGSTS_Q, Smem_tile_ldgsts_q, Smem_tile_tma_q>;

    // The global memory tile to load K.
    // Hopefully we don't need to specialize for hopper.
    using Gmem_tile_ldgsts_k = Gmem_tile_qkv_<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_B, S, D, D,
        true, // use ldgsts
        HEADS_INTERLEAVED, NUM_QKV_MATS, SLIDING_WINDOW_ATTENTION>;

    // The global memory tile to load K with TMA.
    using Gmem_tile_tma_k = Gmem_tile_tma_qkv_<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_B, S, D, false,
        HEADS_INTERLEAVED, NUM_QKV_MATS>;

    // Do we use ldgsts gmem tile or tma gmem tile?
    using Gmem_tile_k = typename std::conditional_t<USE_LDGSTS_K, Gmem_tile_ldgsts_k, Gmem_tile_tma_k>;

    // 1 buffers for K
    enum
    {
        BUFFERS_PER_SMEM_TILE_K = 1
    };

    // K is column major
    using K_layout = fmha::Col;

    // We know K is column-major. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_K
        = Cta_tile_p::K * sizeof(typename Traits_p::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    // The shared memory tile to swizzle K.
    using Smem_tile_ldgsts_k = fmha::Smem_tile_hopper_b<Traits_p, Cta_tile_p, K_layout, 16, BUFFERS_PER_SMEM_TILE_K,
        GMMA_DESC_MODE_K, USE_TMA_K>;

    using Smem_tile_tma_k = fmha::wip::Smem_tile_hopper_b<Traits_p, Cta_tile_p, K_layout, BUFFERS_PER_SMEM_TILE_K,
        GMMA_DESC_MODE_K, GMMA_K_RF, USE_LDGSTS_K>;

    using Smem_tile_k = typename std::conditional_t<USE_LDGSTS_K, Smem_tile_ldgsts_k, Smem_tile_tma_k>;

    // The global memory tile to load V.
    using Gmem_tile_ldgsts_v = Gmem_tile_qkv_<Traits_o, Cta_tile_o, Traits_o::BITS_PER_ELEMENT_B, S, D, D,
        true, // use ldgsts
        HEADS_INTERLEAVED, NUM_QKV_MATS, SLIDING_WINDOW_ATTENTION>;

    // The global memory tile to load V with TMA.
    using Gmem_tile_tma_v = Gmem_tile_tma_qkv_<Traits_o, Cta_tile_o, Traits_o::BITS_PER_ELEMENT_B, S, D, false,
        HEADS_INTERLEAVED, NUM_QKV_MATS>;

    // Do we use ldgsts gmem tile or tma gmem tile?
    using Gmem_tile_v = typename std::conditional_t<USE_LDGSTS_V, Gmem_tile_ldgsts_v, Gmem_tile_tma_v>;

    // 1 buffers for V
    enum
    {
        BUFFERS_PER_SMEM_TILE_V = 1
    };

    // V is row major
    using V_layout = fmha::Row;

    // We know V is row marjor. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_V
        = Cta_tile_o::N * sizeof(typename Traits_o::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    // The shared memory tile to swizzle V.
    using Smem_tile_ldgsts_v = fmha::Smem_tile_v<Traits_o, Cta_tile_o,
        1,    // BUFFERS_PER_TILE
        GMMA_DESC_MODE_V,
        false // USE_TMA_V
        >;

    using Smem_tile_tma_v = fmha::wip::Smem_tile_hopper_b<Traits_o, Cta_tile_o, V_layout, BUFFERS_PER_SMEM_TILE_V,
        GMMA_DESC_MODE_V, GMMA_V_RF, USE_LDGSTS_V>;

    using Smem_tile_v = typename std::conditional_t<USE_LDGSTS_V, Smem_tile_ldgsts_v, Smem_tile_tma_v>;

    // The global memory tile to store O.
    // using Gmem_tile_o = fmha::Gmem_tile_o_hopper<Traits_o, Cta_tile_o>;
    using Gmem_tile_o = fmha::v2::Gmem_tile_o<Traits_o, Cta_tile_o, 1>;

    using Smem_tile_o_ = fmha::Smem_tile_o<Traits_o, Cta_tile_o>;
    static constexpr bool NEEDS_SPLIT_K = WARPS_N > 1;
    using Smem_tile_o = typename std::conditional_t<NEEDS_SPLIT_K, Smem_tile_o_, fmha::Smem_tile_o_dummy>;

    // The amount of shared memory needed to load Q and K.
    enum
    {
        BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE
    };

    // The extra amount of shared memory needed to load V.
    enum
    {
        BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE
    };

    // The amount of shared memory needed for Q, K and V..
    enum
    {
        BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V
    };

    // The amount of shared memory needed to load Q and store O.
    // enum { BYTES_PER_SMEM_QO = Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE };
    // For now let's pretend no smem for O matrix. [Timmy]
    enum
    {
        BYTES_PER_SMEM_QO = Smem_tile_q::BYTES_PER_TILE
    };

    // The amount of over allocated smem to guarantee 1024B alignment.
    enum
    {
        BYTES_FOR_ALIGNMENT = 1024
    };

    // The size in bytes for each SMEM barrier
    enum
    {
        BYTES_PER_SMEM_BARRIER = 8
    };

    // The amount of smem used by smem barrier. Only needed if TMA is used.
    enum
    {
        BYTES_FOR_SMEM_BARRIER_Q = USE_LDGSTS_Q == 1 ? 0 : BUFFERS_PER_SMEM_TILE_Q * BYTES_PER_SMEM_BARRIER
    };

    // The amount of smem used by smem barrier. Only needed if TMA is used.
    // each smem barrier is 8 bytes, each buffer has 2 barriers
    enum
    {
        BYTES_FOR_SMEM_BARRIER_K = USE_LDGSTS_K == 1 ? 0 : BUFFERS_PER_SMEM_TILE_K * BYTES_PER_SMEM_BARRIER
    };

    // The amount of smem used by smem barrier. Only needed if TMA is used.
    // Currently, K and V can share the same barrier.
    enum
    {
        BYTES_FOR_SMEM_BARRIER_V = 0
    };

    // The amount of smem used by smem barrier. Only needed if TMA is used.
    enum
    {
        BYTES_FOR_SMEM_BARRIER = BYTES_FOR_SMEM_BARRIER_Q + BYTES_FOR_SMEM_BARRIER_K + BYTES_FOR_SMEM_BARRIER_V
    };

    // TODO move those
    enum
    {
        BYTES_FOR_SOFTMAX = WARPS_N == 1 ? 0 : sizeof(float) * WARPS_N * 64
    };

    enum
    {
        BYTES_PER_SMEM_O = WARPS_N == 1 ? 0 : WARPS_N * 64 * D * sizeof(typename Traits_o::Epilogue_type)
    };

    static_assert(Smem_tile_o::BYTES_PER_TILE == (int) BYTES_PER_SMEM_O);

    // The amount of shared memory needed for Q, K, V and O.
    // TODO double check.
    // - For GMMA QKV are always stored in SMEM.
    // - Cannot share SMEM K/V
    // - O needs to be separate
    // enum { BYTES_PER_SMEM = fmha::Max<BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO>::VALUE
    enum
    {
        BYTES_PER_SMEM
        = BYTES_PER_SMEM_QKV + BYTES_PER_SMEM_O + BYTES_FOR_SOFTMAX + BYTES_FOR_SMEM_BARRIER + BYTES_FOR_ALIGNMENT
    };

    // The number of threads.
    enum
    {
        THREADS = Cta_tile_p::THREADS_PER_CTA
    };

    // Make sure the number of threads matches both CTAs.
    static_assert((int) THREADS == (int) Cta_tile_o::THREADS_PER_CTA, "");

    // The compute tile for P = Q*K.
    using Compute_tile_p = fmha::Compute_tile_with_gmma<Traits_p, Cta_tile_p, Smem_tile_q, Smem_tile_k,
        Traits_p::GMMA_A_RF, Traits_p::GMMA_B_RF>;
    // The compute tile for O = S*V.
    using Compute_tile_o = fmha::Compute_tile_with_gmma<Traits_o,
        // TODO TMA path?
        Cta_tile_o,
        // typename Smem_tile_v::Cta_tile_gmma,
        Smem_tile_q, // we don't need to pass smem_tile here? Not really.
        Smem_tile_v, Traits_o::GMMA_A_RF, Traits_o::GMMA_B_RF>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The BMM1 instruction traits.
    typename Traits_p,
    // The BMM2 instruction traits.
    typename Traits_o,
    // The sequence length.
    int S,
    // The hidden size per head.
    int D,
    // The number of timesteps per iteration of the main loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The attention mask type (2 denotes dense mask, 3 denotes causal mask).
    int MASK_VERSION,
    // The flags.
    uint32_t FLAGS = 0x8>
using FMHA_kernel_traits_hopper_v2 = FMHA_kernel_traits_hopper<Traits_p, Traits_o,
    fmha::v2::Gmem_tile_qkv, // hopefully we don't need to specialize for hopper ldgsts
    fmha::v2::Gmem_tile_tma_qkv, fmha::v2::Gmem_tile_o, S, D, STEP, WARPS_M, WARPS_N, 2, MASK_VERSION, FLAGS>;

////////////////////////////////////////////////////////////////////////////////////////////////////
