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

#include "fmha/numeric_types.h"
#include <fmha/utils.h>

#define FMHA_DIV_UP(m, n) (((m) + (n) -1) / (n))

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Trait class for heuristically determining the tile sizes
template <bool GRANULAR, int STEP, int S, int D, int DV, int K_PER_MMA>
struct Traits_tile_size;

template <int STEP, int S, int D, int DV, int K_PER_MMA>
struct Traits_tile_size</* GRANULAR = */ false, STEP, S, D, DV, K_PER_MMA>
{
    enum
    {
        CTA_P_TILE_M = STEP,
        CTA_P_TILE_N = S,
        CTA_P_TILE_K = D,
        CTA_O_TILE_M = CTA_P_TILE_M,
        CTA_O_TILE_N = DV,
        CTA_O_TILE_K = S
    };
};

template <int STEP, int S, int D, int DV, int K_PER_MMA>
struct Traits_tile_size</* GRANULAR = */ true, STEP, S, D, DV, K_PER_MMA>
{
    enum
    {
        CTA_P_TILE_M = STEP,
        CTA_P_TILE_N = S,
        // D =16: CTA_P_TILE_K=16
        // D =32: CTA_P_TILE_K=32
        // D>=64: CTA_P_TILE_K=64
        CTA_P_TILE_K = D < 32 ? 16 : (D < 64 ? 32 : 64),
        CTA_O_TILE_M = CTA_P_TILE_M,
        // D =512: CTA_TILE_N=256
        // D<=256: CTA_TILE_N=D
        CTA_O_TILE_N = DV > 256 ? 256 : DV,
        // D =512: CTA_O_TILE_K=16
        // D =256: CTA_O_TILE_K=32
        // D<=128: CTA_O_TILE_K=64
        CTA_O_TILE_K = std::max(K_PER_MMA, DV > 256 ? 16 : (DV > 128 ? 32 : 64))
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The GPU architecture.
    typename Gpu_arch,
    // The number of rows in the CTA tile.
    int M_,
    // The number of cols in the CTA tile.
    int N_,
    // The number of elements in the the K dimension of the GEMM loop.
    int K_,
    // The number of valid cols in the CTA tile.
    int VALID_N_,
    // The number of valid elements in the the K dimension of the GEMM loop.
    int VALID_K_,
    // The number of rows of warps.
    int WARPS_M_,
    // The number of cols of warps.
    int WARPS_N_,
    // The number of warps in the K dimension of the GEMM loop.
    int WARPS_K_>
struct Cta_tile_
{

    enum
    {
        M = M_,
        N = N_,
        K = K_,
        VALID_N = VALID_N_,
        VALID_K = VALID_K_
    };

    // The number of warps.
    enum
    {
        WARPS_M = WARPS_M_,
        WARPS_N = WARPS_N_,
        WARPS_K = WARPS_K_
    };

    // The number of warps per CTA.
    enum
    {
        WARPS_PER_CTA = WARPS_M * WARPS_N * WARPS_K
    };

    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = Gpu_arch::THREADS_PER_WARP
    };

    // The number of threads per CTA.
    enum
    {
        THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The GPU architecture.
    typename Gpu_arch_,
    // The type of the elements of A.
    typename A_type_,
    // The type of the elements of B.
    typename B_type_,
    // The type of the elements of C.
    typename C_type_,
    // The type of the elements of the accumulators.
    typename Accumulator_type_,
    // The type of the elements of the epilogue.
    typename Epilogue_type_>
struct Traits
{

    // The architecture.
    using Gpu_arch = Gpu_arch_;
    // The data type for A elements.
    using A_type = A_type_;
    // The data type for B elements.
    using B_type = B_type_;
    // The data type for C elements.
    using C_type = C_type_;
    // The data type for accumulators.
    using Accumulator_type = Accumulator_type_;
    // The data type of the math in the epilogue.
    using Epilogue_type = Epilogue_type_;

    // Create the description of the CTA tile from a configuration.
    template <int M, int N, int K, int VALID_N, int VALID_K, int WARPS_M, int WARPS_N, int WARPS_K>
    using Cta_tile_extd = Cta_tile_<Gpu_arch, M, N, K, VALID_N, VALID_K, WARPS_M, WARPS_N, WARPS_K>;

    // The number of bits per element of A.
    enum
    {
        BITS_PER_ELEMENT_A = sizeof(A_type) * 8
    };

    // An offset in bytes for A.
    static inline __host__ __device__ int64_t offset_in_bytes_a(int64_t offset)
    {
        return offset * static_cast<int64_t>(sizeof(A_type));
    }

    // The number of bits per element of B.
    enum
    {
        BITS_PER_ELEMENT_B = sizeof(B_type) * 8
    };

    // An offset in bytes for B.
    static inline __host__ __device__ int64_t offset_in_bytes_b(int64_t offset)
    {
        return offset * static_cast<int64_t>(sizeof(B_type));
    }

    // The number of bits per element of C.
    enum
    {
        BITS_PER_ELEMENT_C = sizeof(C_type) * 8
    };

    // An offset in bytes for C.
    static inline __host__ __device__ int64_t offset_in_bytes_c(int64_t offset)
    {
        return offset * static_cast<int64_t>(sizeof(C_type));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Gpu_arch_base
{

    // By default, architectures have 32 threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };

    // By default, architectures do not support LDGSTS.
    enum
    {
        HAS_LDGSTS = 0
    };

    // By default, architecture do not support super HMMA
    enum
    {
        HAS_SUPER_HMMA = 0
    };

    // By default, architecture do not support TMA
    enum
    {
        HAS_TMA = 0
    };

    // By default, architecture do not support GMMA
    enum
    {
        HAS_GMMA = 0
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_, typename Cta_tile_>
using Cta_tile_with_k_with_padding = typename Traits_::template Cta_tile_extd<Cta_tile_::M, Cta_tile_::N,
    Next_power_of_two<Cta_tile_::K>::VALUE, Cta_tile_::N, Next_power_of_two<Cta_tile_::K>::VALUE, Cta_tile_::WARPS_M,
    Cta_tile_::WARPS_N, Cta_tile_::WARPS_K>;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta : public Gpu_arch_base
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int N_PER_MMA_ = 16, int K_PER_MMA_ = 8>
struct Volta_mma_tile
{

    // The number of elements computed with a single warp-MMA.
    enum
    {
        M_PER_MMA = 16,
        N_PER_MMA = N_PER_MMA_,
        K_PER_MMA = K_PER_MMA_
    };

    // The number of elements computed with a single CTA-MMA.
    enum
    {
        M_PER_MMA_PER_CTA = M_PER_MMA * Cta_tile::WARPS_M,
        N_PER_MMA_PER_CTA = N_PER_MMA * Cta_tile::WARPS_N,
        K_PER_MMA_PER_CTA = K_PER_MMA * Cta_tile::WARPS_K
    };

    // The number of MMAs needed to compute the GEMM.
    enum
    {
        MMAS_M = (Cta_tile::M + M_PER_MMA_PER_CTA - 1) / M_PER_MMA_PER_CTA,
        MMAS_N = (Cta_tile::N + N_PER_MMA_PER_CTA - 1) / N_PER_MMA_PER_CTA,
        MMAS_K = (Cta_tile::K + K_PER_MMA_PER_CTA - 1) / K_PER_MMA_PER_CTA
    };

    // The number of valid MMAs (for Head Size)
    enum
    {
        // tile o
        VALID_MMAS_N = Div_up<Cta_tile::VALID_N, N_PER_MMA_PER_CTA>::VALUE,
        // tile p
        VALID_MMAS_K = Div_up<Cta_tile::VALID_K, K_PER_MMA_PER_CTA>::VALUE,
    };

    // The number of elements computed per warp.
    enum
    {
        M_PER_WARP = MMAS_M * M_PER_MMA,
        N_PER_WARP = MMAS_N * N_PER_MMA,
        K_PER_WARP = MMAS_K * K_PER_MMA,
    };

    // Do we enable the fast path for LDS.
    enum
    {
        ENABLE_LDS_FAST_PATH = 0
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_hmma_fp16_traits : public Traits<Volta, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>
{

    // The K_PER_MMA for Volta_hmma_fp16_traits is 8.
    enum
    {
        K_PER_MMA = 8
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Volta_mma_tile<Cta_tile, 16, K_PER_MMA>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_hmma_fp16_16x16x16_traits : public Traits<Volta, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>
{

    // The K_PER_MMA for Volta_hmma_fp16_16x16x16_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Volta_mma_tile<Cta_tile, 16, K_PER_MMA>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Volta_imma_int8_int32_traits : public Traits<Volta, int8_t, int8_t, int8_t, int32_t, float>
{

    // The K_PER_MMA for Volta_imma_int8_int32_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Volta_mma_tile<Cta_tile, 16, K_PER_MMA>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing : public Gpu_arch_base
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int K_PER_MMA_>
struct Turing_mma_tile
{

    // The number of elements computed with a single warp-MMA.
    enum
    {
        M_PER_MMA = 16,
        N_PER_MMA = 16,
        K_PER_MMA = K_PER_MMA_
    };

    // The number of elements computed with a single CTA-MMA.
    enum
    {
        M_PER_MMA_PER_CTA = M_PER_MMA * Cta_tile::WARPS_M,
        N_PER_MMA_PER_CTA = N_PER_MMA * Cta_tile::WARPS_N,
        K_PER_MMA_PER_CTA = K_PER_MMA * Cta_tile::WARPS_K
    };

    // The number of MMAs needed to compute the GEMM.
    enum
    {
        MMAS_M = Div_up<Cta_tile::M, M_PER_MMA_PER_CTA>::VALUE,
        MMAS_N = Div_up<Cta_tile::N, N_PER_MMA_PER_CTA>::VALUE,
        MMAS_K = Div_up<Cta_tile::K, K_PER_MMA_PER_CTA>::VALUE,
    };

    // The number of valid MMAs (for Head Size)
    enum
    {
        // tile o
        VALID_MMAS_N = Div_up<Cta_tile::VALID_N, N_PER_MMA_PER_CTA>::VALUE,
        // tile p
        VALID_MMAS_K = Div_up<Cta_tile::VALID_K, K_PER_MMA_PER_CTA>::VALUE,
    };

    // The number of elements computed per warp.
    enum
    {
        M_PER_WARP = MMAS_M * M_PER_MMA,
        N_PER_WARP = MMAS_N * N_PER_MMA,
        K_PER_WARP = MMAS_K * K_PER_MMA,
    };

    // The distribution of threads in the output tile.
    enum
    {
        THREADS_PER_MMA_M = 8,
        THREADS_PER_MMA_N = 4,
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Turing_hmma_tile : public Turing_mma_tile<Cta_tile, 8>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_hmma_fp16_traits : public Traits<Turing, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>
{

    // The K_PER_MMA for Turing_hmma_fp16_traits is 8.
    enum
    {
        K_PER_MMA = 8
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Turing_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_hmma_fp32_traits : public Traits<Turing, uint16_t, uint16_t, uint16_t, float, float>
{

    // The K_PER_MMA for Turing_hmma_fp32_traits is 8.
    enum
    {
        K_PER_MMA = 8
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Turing_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Turing_imma_int8_tile : public Turing_mma_tile<Cta_tile, 16>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Turing_imma_int8_int32_traits : public Traits<Turing, int8_t, int8_t, int8_t, int32_t, float>
{

    // The K_PER_MMA for Turing_imma_int8_int32_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Turing_imma_int8_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere : public Gpu_arch_base
{
    // It has LDGSTS.
    enum
    {
        HAS_LDGSTS = 1
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int K_PER_MMA = 16>
struct Ampere_hmma_tile : public Turing_mma_tile<Cta_tile, K_PER_MMA>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_hmma_fp16_traits : public Traits<Ampere, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>
{

    // The K_PER_MMA for Ampere_hmma_fp16_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_hmma_fp32_traits : public Traits<Ampere, uint16_t, uint16_t, uint16_t, float, uint16_t>
{

    // The K_PER_MMA for Ampere_hmma_fp32_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// used for Epilogue_type = bf16_t (similar to Ampere_hmma_fp16_traits).
struct Ampere_hmma_bf16_bf16_traits : public Traits<Ampere, bf16_t, bf16_t, bf16_t, bf16_t, bf16_t>
{

    // The K_PER_MMA for Ampere_hmma_bf16_bf16_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_hmma_bf16_traits : public Traits<Ampere, bf16_t, bf16_t, bf16_t, float, bf16_t>
{

    // The K_PER_MMA for Ampere_hmma_bf16_traits is 16.
    enum
    {
        K_PER_MMA = 16
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ampere_hmma_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Ampere_imma_int8_tile : public Turing_mma_tile<Cta_tile, 32>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ampere_imma_int8_int32_traits : public Traits<Ampere, int8_t, int8_t, int8_t, int32_t, float>
{

    // The K_PER_MMA for Ampere_imma_int8_int32_traits is 32.
    enum
    {
        K_PER_MMA = 32
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ampere_imma_int8_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ada : public Gpu_arch_base
{
    // It has LDGSTS.
    enum
    {
        HAS_LDGSTS = 1
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following partial traits are mapped to Ampere_hmma_fp16_traits in fmha/kernel_traits.h.
//
// It is easier to implement setup.py this way.
struct Ada_hmma_fp16_traits
{
};

struct Ada_hmma_fp32_traits
{
};

struct Ada_imma_int8_int32_traits
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Ada_qmma_fp8_tile : public Turing_mma_tile<Cta_tile, 32>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ada_qmma_e4m3_fp16_traits : public Traits<Ada, e4m3_t, e4m3_t, e4m3_t, uint16_t, uint16_t>
{

    // The K_PER_MMA for Ada_qmma_e4m3_fp16_traits is 32.
    enum
    {
        K_PER_MMA = 32
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ada_qmma_fp8_tile<Cta_tile>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ada_qmma_e4m3_fp32_traits : public Traits<Ada, e4m3_t, e4m3_t, e4m3_t, float, float>
{

    // The K_PER_MMA for Ada_qmma_e4m3_fp32_traits is 32.
    enum
    {
        K_PER_MMA = 32
    };

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Ada_qmma_fp8_tile<Cta_tile>;

    static constexpr float SOFTMAX_FP_QUANT_SCALE = Softmax_fp_quant_scale<Traits::A_type>();
    static constexpr float SOFTMAX_FP_DEQUANT_SCALE = 1.f / SOFTMAX_FP_QUANT_SCALE;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hopper : public Gpu_arch_base
{
    // It has LDGSTS.
    enum
    {
        HAS_LDGSTS = 1
    };

    // It has TMA.
    enum
    {
        HAS_TMA = 1
    };

    // It has GMMA
    enum
    {
        HAS_GMMA = 1
    };

    // for Hopper there are 4 warps per warpgroup.
    enum
    {
        WARPS_PER_WARP_GROUP = 4
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Hopper related code.
// SHOULD we move this to a different file??
////////////////////////////////////////////////////////////////////////////////////////////////////
template <int HEIGHT_ = 1, int WIDTH_ = 1, int DEPTH_ = 1>
struct Hopper_cga_tile
{

    // The size of the CGA in terms of CTA
    enum
    {
        CLUSTER_HEIGHT = HEIGHT_
    };

    enum
    {
        CLUSTER_WIDTH = WIDTH_
    };

    enum
    {
        CLUSTER_DEPTH = DEPTH_
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gpu_arch_,
    int M_,            // CTA tile M
    int N_,            // CTA tile N
    int K_,            // CTA tile K
    int VALID_N_,      // CTA tile valid N
    int VALID_K_,      // CTA tile valid K
    int WARP_GROUP_M_, // Number of warp group along M dim
    int WARP_GROUP_N_, // Number of warp group along N dim
    int WARP_GROUP_K_> // Number of warp group along K dim
struct Hopper_cta_tile
{
    // GPU arch.
    using Gpu_arch = Gpu_arch_;

    // The size of the CTA tile.
    // TODO: support D (not power of 2)
    enum
    {
        M = M_,
        N = N_,
        K = K_,
        VALID_N = VALID_N_,
        VALID_K = VALID_K_
    };

    // The number of warp groups.
    enum
    {
        WARP_GROUP_M = WARP_GROUP_M_,
        WARP_GROUP_N = WARP_GROUP_N_,
        WARP_GROUP_K = WARP_GROUP_K_
    };

    // The number of warps in a warp group.
    enum
    {
        WARPS_M_PER_GROUP = 4,
        WARPS_N_PER_GROUP = 1,
        WARPS_K_PER_GROUP = 1,
    };

    // The number of warps in a cta.
    enum
    {
        WARPS_M = WARPS_M_PER_GROUP * WARP_GROUP_M_,
        WARPS_N = WARPS_N_PER_GROUP * WARP_GROUP_N_,
        WARPS_K = WARPS_K_PER_GROUP * WARP_GROUP_K_
    };

    // The number of warps per CTA.
    enum
    {
        WARPS_PER_CTA = WARP_GROUP_M * WARP_GROUP_N * WARP_GROUP_K * Gpu_arch::WARPS_PER_WARP_GROUP
    };

    // The number of warps per warpgroup.
    enum
    {
        WARPS_PER_WARP_GROUP = Gpu_arch::WARPS_PER_WARP_GROUP
    };

    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = Gpu_arch::THREADS_PER_WARP
    };

    // the number of threads per warpgroup.
    enum
    {
        THREADS_PER_WARP_GROUP = THREADS_PER_WARP * WARPS_PER_WARP_GROUP
    };

    // The number of threads per CTA.
    enum
    {
        THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP
    };

    enum
    {
        GROUPS_M = 1
    };

    enum
    {
        GROUPS_N = 1
    };

    enum
    {
        GROUPS_K = 1
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, int GMMA_M, int GMMA_N, int GMMA_K>
struct Hopper_gmma_tile
{

    // The number of elements computed with a single warp group mma.
    enum
    {
        M_PER_MMA = GMMA_M,
        N_PER_MMA = GMMA_N,
        K_PER_MMA = GMMA_K
    };

    // The number of warp groups.
    enum
    {
        NUM_WARP_GROUPS = Cta_tile::WARP_GROUP_M * Cta_tile::WARP_GROUP_N * Cta_tile::WARP_GROUP_K
    };

    // The number of elements computed with a single CTA-MMA.
    enum
    {
        M_PER_MMA_PER_CTA = M_PER_MMA * Cta_tile::WARP_GROUP_M,
        N_PER_MMA_PER_CTA = N_PER_MMA * Cta_tile::WARP_GROUP_N,
        K_PER_MMA_PER_CTA = K_PER_MMA * Cta_tile::WARP_GROUP_K
    };

    // The number of MMAs needed to compute the GEMM.
    enum
    {
        MMAS_M = (Cta_tile::M + M_PER_MMA_PER_CTA - 1) / M_PER_MMA_PER_CTA,
        MMAS_N = (Cta_tile::N + N_PER_MMA_PER_CTA - 1) / N_PER_MMA_PER_CTA,
        MMAS_K = (Cta_tile::K + K_PER_MMA_PER_CTA - 1) / K_PER_MMA_PER_CTA,
    };

    // The number of valid MMAs (for Head Size)
    enum
    {
        // tile o
        VALID_MMAS_N = Div_up<Cta_tile::VALID_N, N_PER_MMA_PER_CTA>::VALUE,
        // tile p
        VALID_MMAS_K = Div_up<Cta_tile::VALID_K, K_PER_MMA_PER_CTA>::VALUE,
    };

    // The number of elements computed per warp group.
    enum
    {
        M_PER_WARP_GROUP = MMAS_M * M_PER_MMA,
        N_PER_WARP_GROUP = MMAS_N * N_PER_MMA,
        K_PER_WARP_GROUP = MMAS_K * K_PER_MMA,
    };

    // the size of GMMA group, which is GMMA_M x GMMA_N x Kblock.
    enum
    {
        M_PER_GMMA_GROUP = GMMA_M,
        N_PER_GMMA_GROUP = GMMA_N,
        K_PER_GMMA_GROUP = Cta_tile::K,
    };

    // The distribution of threads in the output tile.
    // TODO
    enum
    {
        THREADS_PER_MMA_M = 8,
        THREADS_PER_MMA_N = 4,
    };

    // The number of core matrices per GMMA.
    enum
    {
        CORES_M_PER_GROUP = 8 * Cta_tile::WARPS_M_PER_GROUP,
        CORES_N_PER_GROUP = 8 * Cta_tile::WARPS_N_PER_GROUP,
        CORES_M = GMMA_M / CORES_M_PER_GROUP,
        CORES_N = GMMA_N / CORES_N_PER_GROUP,
    };

    // The number of logical rows/cols per thread.
    enum
    {
        // A thread owns 1 row per core matrix.
        ROWS_PER_THREAD = CORES_M,
        // A thread owns 2 col per core matrix.
        COLS_PER_THREAD = CORES_N * 2,
    };

    static_assert(ROWS_PER_THREAD == 2);
    static_assert(COLS_PER_THREAD == GMMA_N / 4);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Hopper_instructions
{
    HGMMA_FP16,
    HGMMA_BF16,
    HGMMA_FP32,
    IGMMA_INT32,
    QGMMA_E4M3_FP32,
    QGMMA_E5M2_FP32,
    QGMMA_E4M3_FP16,
    QGMMA_E5M2_FP16
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Hopper HGMMA FP16 Traits
template <int GMMA_M_, // GMMA instruction shape in M dim
    int GMMA_N_,       // GMMA instruction shape in N dim
    int GMMA_K_,       // GMMA instruction shape in K dim
    bool GMMA_A_RF_,   // GMMA A operand coming from RF?
    bool GMMA_B_RF_    // GMMA B operand coming from RF?
    >
struct Hopper_hgmma_fp16_traits : public Traits<Hopper, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>
{
    // The GMMA shape.
    enum
    {
        GMMA_M = GMMA_M_,
        GMMA_N = GMMA_N_,
        GMMA_K = 16
    };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirements.
    static_assert(GMMA_K == 16, "GMMA K must be 16; this might change");
    static_assert(GMMA_M == 64, "GMMA M must be 64; this might change");
    static_assert(GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change");
    static_assert(GMMA_N <= 256, "GMMA N must be no larger than 256; this might change");

    // GMMA does not allow both operands coming from RF.
    static_assert((GMMA_A_RF && GMMA_B_RF) != true, "GMMA does not allow both operands coming from RF.");

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The Cta tile.
    template <int M, int N, int K, int VALID_N, int VALID_K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_padded_tile = Hopper_cta_tile<Hopper, M, N, K, VALID_N, VALID_K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The CGA Tile
    template <int HEIGHT = 1, int WIDTH = 1, int DEPTH = 1>
    using Cga_tile = Hopper_cga_tile<HEIGHT, WIDTH, DEPTH>;

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Hopper_gmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;

    // The handle to differentiate instructions.
    static constexpr fmha::Hopper_instructions HOPPER_INSTRUCTION = fmha::Hopper_instructions::HGMMA_FP16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Hopper HGMMA FP32 Traits
template <int GMMA_M_, // GMMA instruction shape in M dim
    int GMMA_N_,       // GMMA instruction shape in N dim
    int GMMA_K_,       // GMMA instruction shape in K dim
    bool GMMA_A_RF_,   // GMMA A operand coming from RF?
    bool GMMA_B_RF_    // GMMA B operand coming from RF?
    >
struct Hopper_hgmma_fp32_traits : public Traits<Hopper, uint16_t, uint16_t, uint16_t, float, uint16_t>
{
    // The GMMA shape.
    enum
    {
        GMMA_M = GMMA_M_,
        GMMA_N = GMMA_N_,
        GMMA_K = 16
    };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirements.
    static_assert(GMMA_K == 16, "GMMA K must be 16; this might change");
    static_assert(GMMA_M == 64, "GMMA M must be 64; this might change");
    static_assert(GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change");
    static_assert(GMMA_N <= 256, "GMMA N must be no larger than 256; this might change");

    // GMMA does not allow both operands coming from RF.
    static_assert((GMMA_A_RF && GMMA_B_RF) != true, "GMMA does not allow both operands coming from RF.");

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The Cta tile.
    template <int M, int N, int K, int VALID_N, int VALID_K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_padded_tile = Hopper_cta_tile<Hopper, M, N, K, VALID_N, VALID_K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The CGA Tile
    template <int HEIGHT = 1, int WIDTH = 1, int DEPTH = 1>
    using Cga_tile = Hopper_cga_tile<HEIGHT, WIDTH, DEPTH>;

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Hopper_gmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;

    // The handle to differentiate instructions.
    static constexpr fmha::Hopper_instructions HOPPER_INSTRUCTION = fmha::Hopper_instructions::HGMMA_FP32;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Hopper BF16 HGMMA Traits
template <int GMMA_M_, // GMMA instruction shape in M dim
    int GMMA_N_,       // GMMA instruction shape in N dim
    int GMMA_K_,       // GMMA instruction shape in K dim
    bool GMMA_A_RF_,   // GMMA A operand coming from RF?
    bool GMMA_B_RF_    // GMMA B operand coming from RF?
    >
struct Hopper_hgmma_bf16_traits : public Traits<Hopper, bf16_t, bf16_t, bf16_t, float, bf16_t>
{
    // The GMMA shape.
    enum
    {
        GMMA_M = GMMA_M_,
        GMMA_N = GMMA_N_,
        GMMA_K = 16
    };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirements.
    static_assert(GMMA_K == 16, "GMMA K must be 16; this might change");
    static_assert(GMMA_M == 64, "GMMA M must be 64; this might change");
    static_assert(GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change");
    static_assert(GMMA_N <= 256, "GMMA N must be no larger than 256; this might change");

    // GMMA does not allow both operands coming from RF.
    static_assert((GMMA_A_RF && GMMA_B_RF) != true, "GMMA does not allow both operands coming from RF.");

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The Cta tile.
    template <int M, int N, int K, int VALID_N, int VALID_K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_padded_tile = Hopper_cta_tile<Hopper, M, N, K, VALID_N, VALID_K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The CGA Tile
    template <int HEIGHT = 1, int WIDTH = 1, int DEPTH = 1>
    using Cga_tile = Hopper_cga_tile<HEIGHT, WIDTH, DEPTH>;

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Hopper_gmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;

    // The handle to differentiate instructions.
    static constexpr fmha::Hopper_instructions HOPPER_INSTRUCTION = fmha::Hopper_instructions::HGMMA_BF16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Hopper IGMMA Traits
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M_, // GMMA instruction shape in M dim
    int GMMA_N_,       // GMMA instruction shape in N dim
    int GMMA_K_,       // GMMA instruction shape in K dim
    bool GMMA_A_RF_,   // GMMA A operand coming from RF?
    bool GMMA_B_RF_    // GMMA B operand coming from RF?
    >
struct Hopper_igmma_int8_int32_traits : public Traits<Hopper, int8_t, int8_t, int8_t, int32_t, float>
{

    using Base = Traits<Hopper, int8_t, int8_t, int8_t, int32_t, float>;

    // The GMMA shape
    enum
    {
        GMMA_M = GMMA_M_
    };

    enum
    {
        GMMA_N = GMMA_N_
    };

    enum
    {
        GMMA_K = 32
    };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirement
    static_assert(GMMA_K == 32, "GMMA K must be 32; this might change");
    static_assert(GMMA_M == 64, "GMMA M must be 64; this might change");
    static_assert(GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change");
    static_assert(GMMA_N <= 256, "GMMA N must be no larger than 256; this might change");

    // GMMA does not allow both operands coming from RF.
    static_assert((GMMA_A_RF && GMMA_B_RF) != true, "GMMA does not allow both operands coming from RF.");

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The Cta tile.
    template <int M, int N, int K, int VALID_N, int VALID_K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_padded_tile = Hopper_cta_tile<Hopper, M, N, K, VALID_N, VALID_K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The CGA Tile
    template <int HEIGHT = 1, int WIDTH = 1, int DEPTH = 1>
    using Cga_tile = Hopper_cga_tile<HEIGHT, WIDTH, DEPTH>;

    // The MMA tile.
    template <typename Cta_tile>
    using Mma_tile = Hopper_gmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Hopper QGMMA Traits
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M_, // GMMA instruction shape in M dim
    int GMMA_N_,       // GMMA instruction shape in N dim
    int GMMA_K_,       // GMMA instruction shape in K dim
    bool GMMA_A_RF_,   // GMMA A operand coming from RF?
    bool GMMA_B_RF_,   // GMMA B operand coming from RF?
    typename Input_type_A_ = e4m3_t, typename Input_type_B_ = e4m3_t, typename Output_type_ = e4m3_t>
struct Hopper_qgmma_fp8_fp32_traits : public Traits<Hopper, Input_type_A_, Input_type_B_, Output_type_, float, float>
{

    using Base = Traits<Hopper, Input_type_A_, Input_type_B_, Output_type_, float, float>;

    using Input_type_A = Input_type_A_;
    using Input_type_B = Input_type_B_;
    using Output_type = Output_type_;

    // The GMMA shape
    enum
    {
        GMMA_M = GMMA_M_
    };

    enum
    {
        GMMA_N = GMMA_N_
    };

    enum
    {
        GMMA_K = 32
    };

    // is A operand in RF for GMMA?
    static constexpr bool GMMA_A_RF = GMMA_A_RF_;

    // is B operand in RF for GMMA?
    static constexpr bool GMMA_B_RF = GMMA_B_RF_;

    // GMMA shape has certain requirement
    static_assert(GMMA_K == 32, "GMMA K must be 32; this might change");
    static_assert(GMMA_M == 64, "GMMA M must be 64; this might change");
    static_assert(GMMA_N % 8 == 0, "GMMA N must be multiple of 8; this might change");
    static_assert(GMMA_N <= 256, "GMMA N must be no larger than 256; this might change");

    // GMMA does not allow both operands coming from RF.
    static_assert((GMMA_A_RF && GMMA_B_RF) != true, "GMMA does not allow both operands coming from RF.");

    // The Cta tile.
    template <int M, int N, int K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_tile = Hopper_cta_tile<Hopper, M, N, K, N, K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The Cta tile.
    template <int M, int N, int K, int VALID_N, int VALID_K, int Warpgroup_M, int Warpgroup_N, int Warpgroup_K>
    using Cta_padded_tile = Hopper_cta_tile<Hopper, M, N, K, VALID_N, VALID_K, Warpgroup_M, Warpgroup_N, Warpgroup_K>;

    // The CGA Tile
    template <int HEIGHT = 1, int WIDTH = 1, int DEPTH = 1>
    using Cga_tile = Hopper_cga_tile<HEIGHT, WIDTH, DEPTH>;

    // The XMMA tile.
    template <typename Cta_tile>
    using Mma_tile = Hopper_gmma_tile<Cta_tile, GMMA_M, GMMA_N, GMMA_K>;

    // Used by low precision floating point types (e4m3, e5m2, etc.)
    static constexpr float SOFTMAX_FP_QUANT_SCALE = Softmax_fp_quant_scale<Input_type_A_>();
    static constexpr float SOFTMAX_FP_DEQUANT_SCALE = 1.f / SOFTMAX_FP_QUANT_SCALE;
};

template <int GMMA_M, // GMMA instruction shape in M dim
    int GMMA_N,       // GMMA instruction shape in N dim
    int GMMA_K,       // GMMA instruction shape in K dim
    bool GMMA_A_RF,   // GMMA A operand coming from RF?
    bool GMMA_B_RF    // GMMA B operand coming from RF?
    >
using Hopper_qgmma_e4m3_fp32_traits
    = Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF, e4m3_t, e4m3_t, e4m3_t>;

} // namespace fmha
