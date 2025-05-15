/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda/std/array>

namespace fmha
{
namespace hopper
{

template <typename Traits_p_, typename Traits_o_, uint32_t S_MAX, uint32_t D, uint32_t STEP_M, uint32_t STEP_N,
    uint32_t WARPS_M, uint32_t WARPS_N, uint32_t FLAGS_>
struct Kernel_traits_fprop
{

    static constexpr int VERSION = 2;

    using Traits_p = Traits_p_;
    using Traits_o = Traits_o_;

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

    // The CTA description for P=QxK'.
    using Cta_tile_p = typename Traits_p::template Cta_tile<STEP_M, STEP_N, D, WARP_GROUP_M, WARP_GROUP_N, 1>;
    // The CTA description for O = S x V.
    using Cta_tile_o = typename Traits_o::template Cta_tile<STEP_M, D, STEP_N, WARP_GROUP_M, 1, WARP_GROUP_N>;

    static constexpr int NUM_MATS = 3;

    // Tile to load from the stacked QKV tensor.
    using Gmem_tile_q = fmha::v2::Gmem_tile_qkv<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP_M, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        NUM_MATS>;

    using Gmem_tile_k = fmha::v2::Gmem_tile_qkv<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP_N, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        NUM_MATS>;

    using Gmem_tile_v = fmha::v2::Gmem_tile_qkv<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP_N, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        NUM_MATS>;

    // TODO
    using Gmem_tile_o = fmha::v2::Gmem_tile_o_gmma_32bit_8bit<Traits_o, Cta_tile_o, 1,
        /*HEADS_INTERLEAVED=*/false>;

    // Number of buffers for Q, dO (fully buffered).
    enum
    {
        BUFFERS_PER_SMEM_TILE_KV = S_MAX / STEP_N
    };

    static_assert(BUFFERS_PER_SMEM_TILE_KV * STEP_N == S_MAX);

    // We know Q is row-major. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_Q
        = Cta_tile_p::K * sizeof(typename Traits_p::A_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    // We know V is column-major. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_K
        = Cta_tile_p::K * sizeof(typename Traits_p::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_D_A
        = Cta_tile_o::K * sizeof(typename Traits_o::A_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_D_B
        = Cta_tile_o::K * sizeof(typename Traits_o::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    template <typename Traits, typename Cta_tile>
    using Smem_tile_a_1buf = fmha::Smem_tile_hopper_a<Traits, Cta_tile, fmha::Row, 16,
        1,     // BUFFERS_PER_SMEM_TILE,
        GMMA_DESC_MODE_D_A,
        false, // USE_TMA_Q,
        Traits::GMMA_A_RF>;

    template <typename Traits, typename Cta_tile>
    using Smem_tile_b_1buf = fmha::Smem_tile_hopper_b<Traits, Cta_tile, fmha::Col, 16,
        1,    // BUFFERS_PER_SMEM_TILE,
        GMMA_DESC_MODE_K,
        false // USE_TMA_K
        >;

    // These A tiles are "dummy" tiles, as currently these GEMMs are A_RF.
    using Smem_tile_s_a = Smem_tile_a_1buf<Traits_o, Cta_tile_o>;

    // A tiles.
    using Smem_tile_q_a = fmha::Smem_tile_hopper_a<Traits_p, Cta_tile_p, fmha::Row, 16,
        2,     // BUFFERS_PER_SMEM_TILE_Q,
        GMMA_DESC_MODE_Q,
        false, // USE_TMA_Q,
        Traits_p::GMMA_A_RF>;

    // B tiles.
    using Smem_tile_kt_b
        = fmha::Smem_tile_hopper_b<Traits_p, Cta_tile_p, fmha::Col, 16, BUFFERS_PER_SMEM_TILE_KV, GMMA_DESC_MODE_K,
            false // USE_TMA_K
            >;
    using Smem_tile_vt_b = Smem_tile_kt_b;
    using Smem_tile_v_b = Smem_tile_vt_b;

    // P = Q x K'
    using Compute_tile_p = fmha::Compute_tile_with_gmma<Traits_p, Cta_tile_p, Smem_tile_q_a, Smem_tile_kt_b,
        Traits_p::GMMA_A_RF, Traits_p::GMMA_B_RF>;

    // o = dP x K
    using Compute_tile_o = fmha::Compute_tile_with_gmma<Traits_o, Cta_tile_o, Smem_tile_s_a, Smem_tile_v_b,
        Traits_o::GMMA_A_RF, Traits_o::GMMA_B_RF>;

    struct Shared_storage
    {

        // Col tiles.
        cuda::std::array<char, Smem_tile_kt_b::BYTES_PER_TILE> kt_b;
        cuda::std::array<char, Smem_tile_vt_b::BYTES_PER_TILE> vt_b;

        // cuda::std::array<char, Smem_tile_v_b ::BYTES_PER_TILE>  v_b;

        // Row tiles. Share with vt
        cuda::std::array<char, Smem_tile_q_a ::BYTES_PER_TILE> q_a;

        // TODO better way?
        cuda::std::array<char, 1024> alignment_;
    };

    static constexpr int THREADS = WARPS_M * WARPS_N * 32;

    static constexpr int SMEM_BYTES = sizeof(Shared_storage);

    static_assert(Cta_tile_p::M == 64);

    // Max 228 KB SMEM per SM.
    static_assert(SMEM_BYTES <= 113 * 1024);
};

template <typename Traits_p_, typename Traits_t_, typename Traits_ds_, typename Traits_dq_, typename Traits_dk_,
    typename Traits_dv_, uint32_t S_MAX, uint32_t D, uint32_t STEP_M, uint32_t STEP_N, uint32_t WARPS_M,
    uint32_t WARPS_N, uint32_t FLAGS_>
struct Kernel_traits_dgrad
{
    static constexpr int VERSION = 2;

    using Traits_p = Traits_p_;
    using Traits_t = Traits_t_;
    using Traits_ds = Traits_ds_;
    using Traits_dq = Traits_dq_;
    using Traits_dk = Traits_dk_;
    using Traits_dv = Traits_dv_;

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

    // The CTA description for P=QxK', dOO=dOxO', dS=dOxV'
    using Cta_tile_p = typename Traits_p ::template Cta_tile<STEP_M, STEP_N, D, WARP_GROUP_M, WARP_GROUP_N, 1>;
    using Cta_tile_t = typename Traits_t ::template Cta_tile<STEP_M, STEP_N, D, WARP_GROUP_M, WARP_GROUP_N, 1>;
    using Cta_tile_ds = typename Traits_ds::template Cta_tile<STEP_M, STEP_N, D, WARP_GROUP_M, WARP_GROUP_N, 1>;
    // dQ = dP  x K (like O = S x V in fprop)
    using Cta_tile_dq = typename Traits_dq::template Cta_tile<STEP_M, D, STEP_N, WARP_GROUP_M, 1, WARP_GROUP_N>;
    // dV = S'  x V
    using Cta_tile_dv = typename Traits_dv::template Cta_tile<STEP_N, D, STEP_M, WARP_GROUP_N, WARP_GROUP_M, 1>;
    // dK = dP' x Q
    using Cta_tile_dk = typename Traits_dk::template Cta_tile<STEP_N, D, STEP_M, WARP_GROUP_N, WARP_GROUP_M, 1>;

    static constexpr int NUM_MATS = 3;

    // Tile to load from the stacked QKV tensor.
    using Gmem_tile_q = fmha::v2::Gmem_tile_qkv<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP_M, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        NUM_MATS>;

    using Gmem_tile_k = fmha::v2::Gmem_tile_qkv<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP_N, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        NUM_MATS>;

    using Gmem_tile_v = fmha::v2::Gmem_tile_qkv<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, STEP_N, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        NUM_MATS>;

    // Tile to load from the output tensor O (or dO).
    using Gmem_tile_o = fmha::v2::Gmem_tile_qkv<Traits_t, Cta_tile_t, Traits_t::BITS_PER_ELEMENT_A, STEP_M, D, D,
        true,  // USE_LDGSTS
        false, // HEADS_INTERLEAVED,
        1      // NUM_QKV_MATS
        >;

    using Gmem_tile_dq = fmha::v2::Gmem_tile_o_gmma_32bit_8bit<Traits_dq, Cta_tile_dq, NUM_MATS,
        /*HEADS_INTERLEAVED=*/false>;

    using Gmem_tile_dk = fmha::v2::Gmem_tile_o_gmma_32bit_8bit<Traits_dk, Cta_tile_dk, NUM_MATS,
        /*HEADS_INTERLEAVED=*/false>;

    using Gmem_tile_dv = fmha::v2::Gmem_tile_o_gmma_32bit_8bit<Traits_dv, Cta_tile_dv, NUM_MATS,
        /*HEADS_INTERLEAVED=*/false>;

    // Number of buffers for Q, dO (fully buffered).
    enum
    {
        BUFFERS_PER_SMEM_TILE_Q = S_MAX / STEP_M
    };

    static_assert(BUFFERS_PER_SMEM_TILE_Q * STEP_M == S_MAX);

    // We know Q is row-major. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_Q
        = Cta_tile_p::K * sizeof(typename Traits_p::A_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    // We know V is column-major. So we can also deduce the descriptor mode.
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_K
        = Cta_tile_p::K * sizeof(typename Traits_p::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_D_A
        = Cta_tile_dv::K * sizeof(typename Traits_dv::A_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                     : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_D_B
        = Cta_tile_dv::K * sizeof(typename Traits_dv::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                     : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    template <typename Traits, typename Cta_tile>
    using Smem_tile_a_1buf = fmha::Smem_tile_hopper_a<Traits, Cta_tile, fmha::Row, 16,
        1,     // BUFFERS_PER_SMEM_TILE,
        GMMA_DESC_MODE_D_A,
        false, // USE_TMA_Q,
        Traits::GMMA_A_RF>;

    template <typename Traits, typename Cta_tile>
    using Smem_tile_b_1buf = fmha::Smem_tile_hopper_b<Traits, Cta_tile, fmha::Col, 16,
        1,    // BUFFERS_PER_SMEM_TILE,
        GMMA_DESC_MODE_K,
        false // USE_TMA_K
        >;

    // These A tiles are "dummy" tiles, as currently these GEMMs are A_RF.
    using Smem_tile_dp_a = Smem_tile_a_1buf<Traits_dq, Cta_tile_dq>;
    using Smem_tile_dpt_a = Smem_tile_a_1buf<Traits_dk, Cta_tile_dk>;
    using Smem_tile_st_a = Smem_tile_a_1buf<Traits_dv, Cta_tile_dv>;

    // A tiles.
    using Smem_tile_q_a
        = fmha::Smem_tile_hopper_a<Traits_p, Cta_tile_p, fmha::Row, 16, BUFFERS_PER_SMEM_TILE_Q, GMMA_DESC_MODE_Q,
            false, // USE_TMA_Q,
            Traits_p::GMMA_A_RF>;

    using Smem_tile_do_a
        = fmha::Smem_tile_hopper_a<Traits_t, Cta_tile_t, fmha::Row, 16, BUFFERS_PER_SMEM_TILE_Q, GMMA_DESC_MODE_Q,
            false, // USE_TMA_Q,
            Traits_t::GMMA_A_RF>;

    // B tiles.
    using Smem_tile_o_b = fmha::Smem_tile_hopper_b<Traits_t, Cta_tile_t, fmha::Col, 16,
        2,    // BUFFERS_PER_SMEM_TILE,
        GMMA_DESC_MODE_K,
        false // USE_TMA_K
        >;
    using Smem_tile_kt_b = Smem_tile_o_b;
    using Smem_tile_vt_b = Smem_tile_o_b;

    using Smem_tile_k_b = Smem_tile_b_1buf<Traits_dq, Cta_tile_dq>;
    using Smem_tile_q_b = Smem_tile_b_1buf<Traits_dk, Cta_tile_dk>;
    using Smem_tile_do_b = Smem_tile_b_1buf<Traits_dv, Cta_tile_dv>;

    // P = Q x K'
    using Compute_tile_p = fmha::Compute_tile_with_gmma<Traits_p, Cta_tile_p, Smem_tile_q_a, Smem_tile_kt_b,
        Traits_p::GMMA_A_RF, Traits_p::GMMA_B_RF>;

    // T = dO x O'
    using Compute_tile_t = fmha::Compute_tile_with_gmma<Traits_t, Cta_tile_t, Smem_tile_do_a, Smem_tile_o_b,
        Traits_t::GMMA_A_RF, Traits_t::GMMA_B_RF>;

    // dS = dO x V'
    using Compute_tile_ds = fmha::Compute_tile_with_gmma<Traits_ds, Cta_tile_ds, Smem_tile_do_a, Smem_tile_vt_b,
        Traits_ds::GMMA_A_RF, Traits_ds::GMMA_B_RF>;

    // dQ = dP x K
    using Compute_tile_dq = fmha::Compute_tile_with_gmma<Traits_dq, Cta_tile_dq, Smem_tile_dp_a, Smem_tile_k_b,
        Traits_dq::GMMA_A_RF, Traits_dq::GMMA_B_RF>;

    // dK = dP' x Q
    using Compute_tile_dk = fmha::Compute_tile_with_gmma<Traits_dk, Cta_tile_dk, Smem_tile_dpt_a, Smem_tile_q_b,
        Traits_dk::GMMA_A_RF, Traits_dk::GMMA_B_RF>;

    // dV = S' x dO
    using Compute_tile_dv = fmha::Compute_tile_with_gmma<Traits_dv, Cta_tile_dv, Smem_tile_st_a, Smem_tile_do_b,
        Traits_dv::GMMA_A_RF, Traits_dv::GMMA_B_RF>;

    struct Shared_storage
    {

        // Col tiles.
        cuda::std::array<char, Smem_tile_kt_b::BYTES_PER_TILE> kt_b;
        cuda::std::array<char, Smem_tile_vt_b::BYTES_PER_TILE> vt_b;

        cuda::std::array<char, Smem_tile_q_b ::BYTES_PER_TILE> q_b;
        cuda::std::array<char, Smem_tile_do_b::BYTES_PER_TILE> do_b;
        cuda::std::array<char, Smem_tile_k_b ::BYTES_PER_TILE> k_b;

        // Row tiles.
        cuda::std::array<char, Smem_tile_q_a ::BYTES_PER_TILE> q_a;
        cuda::std::array<char, Smem_tile_do_a::BYTES_PER_TILE> do_a;
        // cuda::std::array<char, Smem_tile_kt_b::BYTES_PER_TILE> ot_b;

        cuda::std::array<float, S_MAX> diag_doo;
        cuda::std::array<fmha::fp16_t, Cta_tile_p::M * Cta_tile_p::N> s_t;
        cuda::std::array<fmha::fp16_t, Cta_tile_p::M * Cta_tile_p::N> dp_t;

        // TODO better way?
        cuda::std::array<char, 1024> alignment_;
    };

    static constexpr int THREADS = WARPS_M * WARPS_N * 32;

    static constexpr int SMEM_BYTES = sizeof(Shared_storage);

    static_assert(Cta_tile_p::M == 64);

    // Max 228 KB SMEM per SM.
    static_assert(SMEM_BYTES <= 113 * 1024);
};

} // namespace hopper
} // namespace fmha
