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

// Target-private copy of the kernel traits of the SM120 e4m3 head-256 context FMHA.
//
// Kernel_traits_ and Kernel_traits_v2_paged_kv_cache from cpp/kernels/fmha_v2/src/fmha/kernel_traits.h, extended with
// USE_EXP2_SOFTMAX (flag 0x4000), STORE_O_FROM_REGS, Gmem_tile_o_from_regs, Q_RESIDENT, the Q_RESIDENT-aware RELOAD_Q
// and BUFFERS_PER_TILE_SMEM_Q, and BYTES_PER_SMEM_QO = 0 with STORE_O_FROM_REGS (the O tiles moved up so
// STORE_O_FROM_REGS can use them). Also: namespace TRTLLM_MV2_NS::fmha_sm120 instead of fmha; the stock helpers
// are qualified (fmha::Traits_reuse, fmha::Traits_o_adapter, fmha::Next_power_of_two, fmha::Traits_tile_size,
// fmha::Gmem_tile_o_dispatcher), and the register-direct O tile is the target copy (fmhaSm120GmemTileO.cuh). The
// global/shared memory tiles, Cta/Mma tiles and instruction traits are the stock fmha ones.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/fmhaSm120GmemTileO.cuh"

#include <fmha/kernel_traits.h>

#include <type_traits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace fmha_sm120
{

// The unqualified stock names in the copy below (Smem tiles, Max, ...) are the fmha ones.
using namespace ::fmha;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // Instruction traits.
    typename Traits_,
    // The global memory tile for Q, K and V.
    template <typename, typename, int, int, int, int, bool, bool, int, bool> class Gmem_tile_q_,
    template <typename, typename, int, int, int, int, bool, bool, int, bool> class Gmem_tile_k_,
    template <typename, typename, int, int, int, int, bool, bool, int, bool> class Gmem_tile_v_,
    // The global memory tile for the output.
    template <typename, typename, int> class Gmem_tile_o_,
    // Sequence length.
    int S,
    // The valid hidden dimension.
    int VALID_D_,
    // The valid hidden dimension of V.
    int VALID_DV_,
    // The iteration step of the outer loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD_,
    // The flags to control the behaviour of LDGs.
    uint32_t FLAGS,
    // The version of the kernel.
    int VERSION_,
    // The mask version of the kernel
    int MASK_VERSION_,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true,
    // non-positive means disabled
    int SAGE_BLOCK_SIZE_Q_ = 0, int SAGE_BLOCK_SIZE_K_ = 0, int SAGE_BLOCK_SIZE_V_ = 0,
    // Enable skip softmax attention feature.
    bool ENABLE_SKIP_SOFTMAX_ = false>
struct Kernel_traits_
{

    // The instruction traits for the Q*K product.
    using Traits_p = typename fmha::Traits_reuse<Traits_>::Traits;
    // The instruction traits for the P*V product. Hack to change the traits for Volta HMMA.
    using Traits_o = typename fmha::Traits_o_adapter<Traits_p, false>::Traits;
    // The instruction traits for the epilogue of the 2nd GEMM. Always use FP16.
    using Traits_e = typename fmha::Traits_o_adapter<Traits_p, BMM2_FP16_EPILOGUE>::Traits;

    // The padded D dimension
    static constexpr int VALID_D = VALID_D_;

    static constexpr int D = fmha::Next_power_of_two<VALID_D>::VALUE;

    static constexpr int VALID_DV = VALID_DV_ > 0 ? VALID_DV_ : VALID_D;

    static constexpr int DV = fmha::Next_power_of_two<VALID_DV>::VALUE;

    static constexpr int SAGE_ATTENTION = SAGE_BLOCK_SIZE_Q_ > 0 || SAGE_BLOCK_SIZE_K_ > 0 || SAGE_BLOCK_SIZE_V_ > 0;

    static constexpr int SAGE_BLOCK_SIZE_Q = SAGE_BLOCK_SIZE_Q_;

    static constexpr int SAGE_BLOCK_SIZE_K = SAGE_BLOCK_SIZE_K_;

    static constexpr int SAGE_BLOCK_SIZE_V = SAGE_BLOCK_SIZE_V_;

    // Are we enabling skip softmax attention feature?
    static constexpr bool ENABLE_SKIP_SOFTMAX = ENABLE_SKIP_SOFTMAX_;

    // TODO: expose these tiling params to the interface
    static constexpr int USE_GRANULAR_TILING = (FLAGS & 0x1000) != 0u; // TODO ANT: check FLAGS

    using Traits_tile_size = fmha::Traits_tile_size<(bool) USE_GRANULAR_TILING, STEP, S, D, DV, Traits_o::K_PER_MMA>;

    static constexpr int CTA_P_TILE_M = Traits_tile_size::CTA_P_TILE_M;

    static constexpr int CTA_P_TILE_N = Traits_tile_size::CTA_P_TILE_N;

    static constexpr int CTA_P_TILE_K = Traits_tile_size::CTA_P_TILE_K;

    static constexpr int CTA_O_TILE_M = Traits_tile_size::CTA_O_TILE_M;

    static constexpr int CTA_O_TILE_N = Traits_tile_size::CTA_O_TILE_N;

    static constexpr int CTA_O_TILE_K = Traits_tile_size::CTA_O_TILE_K;

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = typename Traits_p::template Cta_tile_extd<CTA_P_TILE_M, CTA_P_TILE_N, CTA_P_TILE_K, S, VALID_D,
        WARPS_M, WARPS_N, 1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = typename Traits_o::template Cta_tile_extd<CTA_O_TILE_M, CTA_O_TILE_N, CTA_O_TILE_K, VALID_DV, S,
        WARPS_M, 1, WARPS_N>;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // Compute the total BMM2_MMAS_K (might not the same as Mma_tile_o::MMAS_K if the granular tiling is used).
    static_assert(S % CTA_O_TILE_K == 0, "");

    static constexpr int TOTAL_BMM2_MMAS_K = Mma_tile_o::MMAS_K * (S / CTA_O_TILE_K);

    // Constraints on the K dimension.
    static_assert(Mma_tile_p::K_PER_MMA <= static_cast<int>(D));
    static_assert(Mma_tile_o::K_PER_MMA <= S);

    // The version.
    static constexpr int VERSION = VERSION_;

    // The mask version: padding (2), causal (3), sliding_window_causal (4), bidirectional_sliding_window (5),
    // custom_mask (6).
    static constexpr int MASK_VERSION = MASK_VERSION_;

    // Whether use causal mask or not.
    static constexpr int CAUSAL_MASK = MASK_VERSION_ == 3 || MASK_VERSION_ == 4;

    // Whether use the sliding window attention or not.
    static constexpr int SLIDING_WINDOW_ATTENTION = MASK_VERSION_ == 4;

    // Whether use the bidirectional sliding window attention or not.
    static constexpr int BIDIRECTIONAL_SLIDING_WINDOW_ATTENTION = MASK_VERSION_ == 5;

    // Whether use the custom mask or not.
    static constexpr int CUSTOM_MASK = MASK_VERSION_ == 6;

    // Do we use LDGSTS for Q, K or V.
    static constexpr int USE_LDGSTS_Q = (FLAGS & 0x1u) != 0u;

    static constexpr int USE_LDGSTS_K = (FLAGS & 0x2u) != 0u;

    static constexpr int USE_LDGSTS_V = (FLAGS & 0x4u) != 0u;

    // Do we use one buffer for K and V.
    static constexpr int SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x8u) != 0u;

    // Do we use the scale max trick.
    static constexpr int USE_SCALE_MAX = (FLAGS & 0x10u) != 0u;

    // Are heads in QKV interleaved, i.e. total x h x 3 x d or total x 3 x h x d.
    static constexpr int HEADS_INTERLEAVED = (FLAGS & 0x20u) == 0u;

    // Keep full K matrix in registers.
    static constexpr int K_IN_REGS = (FLAGS & 0x40) == 0u;

    // Do we use only 2 fragments or full fragments for frag_q/k (only used by flash attention)
    static constexpr int LIMIT_QK_FRAGMENTS = ((FLAGS & 0x80u) != 0u && !SHARE_SMEM_FOR_K_AND_V);

    // Do we use only 2 fragments or full fragments for frag_v (only used by flash attention)
    static constexpr int LIMIT_V_FRAGMENTS = ((FLAGS & 0x100u) != 0u && !SHARE_SMEM_FOR_K_AND_V);

    // Limiting QK fragments implies SMEM_K has to reside in SMEM
    static_assert(!(LIMIT_QK_FRAGMENTS && SHARE_SMEM_FOR_K_AND_V), "");

    // Indicates that kernel does not loop over Q tensor, usually kernel name has _nl suffix
    static constexpr int NO_LOOP = (FLAGS & 0x200u) != 0u;

    // Are sequences in one batch interleaved. i.e. s x b x ..., or b x s x ...
    static constexpr int SEQUENCES_INTERLEAVED = (FLAGS & 0x400) != 0u;

    // Use BMM1 softcapping scale or not.
    static constexpr int ENABLE_BMM1_SOFTCAPPING_SCALE = (FLAGS & 0x800) != 0u;

    // Use MTP (multi-token prediction for MLA kernels) or not.
    static constexpr int IS_MTP = (FLAGS & 0x2000) != 0u;

    // Take the softmax in base 2 with ex2.approx (only honored by the SM120 Ada QMMA e4m3 kernels that store O from
    // registers, see Softmax_qmma<Ada_qmma_e4m3_fp32_traits> in fmha/softmax.h). Never set by the generated kernels.
    static constexpr bool USE_EXP2_SOFTMAX = (FLAGS & 0x4000u) != 0u;

    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    static constexpr int CTAS_PER_HEAD = CTAS_PER_HEAD_;

    // The global memory tile to store O.
    using Gmem_tile_o = Gmem_tile_o_<Traits_e, Cta_tile_o, CTAS_PER_HEAD>;
    // The shared memory tile for O.
    using Smem_tile_o = fmha::Smem_tile_o<Traits_e, Cta_tile_o>;

    // Make sure the number of threads match.
    static_assert((int) Gmem_tile_o::THREADS_PER_ROW == (int) Smem_tile_o::THREADS_PER_ROW, "");

    // The register-direct variant of Gmem_tile_o (if any).
    using Gmem_tile_o_from_regs_selector = fmha_sm120::Gmem_tile_o_from_regs_selector<Gmem_tile_o>;

    // Do we store O straight from the accumulators, without staging it in shared memory? Only in the tiled
    // no-loop flash attention kernel (device_flash_attention_nl_tiled) with QMMA FP32 accumulators, 16-bit
    // output and a 64 x 256 O tile on 4x1x1 warps (the SM120 FP8 head-256 kernels): there the staging only
    // permutes data, and its FP32 tile (64 KB) would set the shared memory size of the kernel.
    static constexpr bool STORE_O_FROM_REGS = NO_LOOP && USE_GRANULAR_TILING && Gmem_tile_o_from_regs_selector::VALUE
        && std::is_same<Traits_e, fmha::Ada_qmma_e4m3_fp32_traits>::value && Cta_tile_o::M == 64 && Cta_tile_o::N == 256
        && Cta_tile_o::VALID_N == Cta_tile_o::N && Cta_tile_o::WARPS_M == 4 && Cta_tile_o::WARPS_N == 1
        && Cta_tile_o::WARPS_K == 1;

    // The global memory tile to store O from the accumulators (only used if STORE_O_FROM_REGS).
    using Gmem_tile_o_from_regs = typename Gmem_tile_o_from_regs_selector::Type;

    // Do we keep the whole Q tile (all D / CTA_P_TILE_K column chunks) resident in shared memory, loading it once
    // (in the first KV step) instead of re-loading it for every KV tile? Only for the STORE_O_FROM_REGS kernels
    // (SM120 FP8 head-256, 64 x 256 Q tile = 4 chunks of 64 x 64 e4m3): 4 Q buffers = 16 KB, total shared memory
    // 49152 B, which keeps 2 CTAs/SM. The same Q bytes reach the same MMAs (bitwise identical output).
    static constexpr bool Q_RESIDENT
        = STORE_O_FROM_REGS && D % CTA_P_TILE_K == 0 && D / CTA_P_TILE_K == 4 && VALID_D == D;

    // Do we need to reload Q due to splitting the D ?
    static constexpr int RELOAD_Q = static_cast<int>(CTA_P_TILE_K) != static_cast<int>(D) && !Q_RESIDENT;

    // The number of shared memory buffers to build a software pipeline for Q, K and V.
    static constexpr int BUFFERS_PER_TILE_SMEM_Q
        = Q_RESIDENT ? D / CTA_P_TILE_K : ((USE_GRANULAR_TILING && D > 64) || (USE_LDGSTS_Q && !NO_LOOP) ? 2 : 1);

    static constexpr int BUFFERS_PER_TILE_SMEM_K = USE_GRANULAR_TILING ? 2 : 1;

    static constexpr int BUFFERS_PER_TILE_SMEM_V = USE_GRANULAR_TILING ? 2 : 1;

    // The global memory tile to load Q.
    using Gmem_tile_q = Gmem_tile_q_<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_A, CTA_P_TILE_M, CTA_P_TILE_K,
        VALID_D, USE_LDGSTS_Q, HEADS_INTERLEAVED,
        3,                       // NUM_MATS
        SLIDING_WINDOW_ATTENTION // Not used.
        >;

    // The shared memory tile to swizzle Q.
    using Smem_tile_q
        = fmha::Smem_tile_a<Traits_p, Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, BUFFERS_PER_TILE_SMEM_Q>;

    // The global memory tile to load K.
    using Gmem_tile_k = Gmem_tile_k_<Traits_p, Cta_tile_p, Traits_p::BITS_PER_ELEMENT_B, CTA_P_TILE_N, CTA_P_TILE_K,
        VALID_D, USE_LDGSTS_K, HEADS_INTERLEAVED,
        3, // NUM_MATS
        SLIDING_WINDOW_ATTENTION>;

    // The shared memory tile to swizzle K.
    using Smem_tile_k
        = fmha::Smem_tile_b<Traits_p, Cta_tile_p, fmha::Col, Gmem_tile_k::BYTES_PER_LDG, BUFFERS_PER_TILE_SMEM_K>;

    // The global memory tile to load V.
    using Gmem_tile_v = Gmem_tile_v_<Traits_o, Cta_tile_o, Traits_o::BITS_PER_ELEMENT_B, CTA_O_TILE_K, CTA_O_TILE_N,
        VALID_DV, USE_LDGSTS_V, HEADS_INTERLEAVED,
        3, // NUM_MATS
        SLIDING_WINDOW_ATTENTION>;

    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v<Traits_o, Cta_tile_o, BUFFERS_PER_TILE_SMEM_V>;

    // The number of threads.
    static constexpr int THREADS = Cta_tile_p::THREADS_PER_CTA;

    // Make sure the number of threads matches both CTAs.
    static_assert((int) THREADS == (int) Cta_tile_o::THREADS_PER_CTA, "");

    // The amount of shared memory needed to load Q and K.
    static constexpr int BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE;

    // The extra amount of shared memory needed to load V.
    static constexpr int BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE;

    // The amount of shared memory needed for Q, K and V..
    static constexpr int BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V;

    // The amount of shared memory needed to load/store O.
    static constexpr int BYTES_PER_SMEM_O = Smem_tile_o::BYTES_PER_TILE;

    // The amount of shared memory needed to load Q and store O (O does not use shared memory when it is
    // stored from the accumulators).
    static constexpr int BYTES_PER_SMEM_QO = STORE_O_FROM_REGS
        ? 0
        : (NO_LOOP ? Smem_tile_o::BYTES_PER_TILE : Smem_tile_q::BYTES_PER_TILE + BYTES_PER_SMEM_O);

    // The amount of shared memory needed for Q, K, V and O.
    static constexpr int BYTES_PER_SMEM = fmha::Max<BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO>::VALUE;

    // Make sure we have enough shared memory.
    static_assert(STORE_O_FROM_REGS
            || (NO_LOOP ? Smem_tile_o::BYTES_PER_TILE : Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE)
                <= BYTES_PER_SMEM,
        "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The sequence length.
    int S,
    // The hidden size per head.
    int D,
    // The hidden dimension of V.
    int DV,
    // The number of timesteps per iteration of the main loop.
    int STEP,
    // The number of vertical warps.
    int WARPS_M,
    // The number of horizontal warps.
    int WARPS_N,
    // The number of CTAs per head for Cta_tile_p; equivalent to BMM1 split-K
    int CTAS_PER_HEAD,
    // The flags.
    uint32_t FLAGS = 0x8,
    // The attention mask version (see src/mask.h).
    int MASK_VERSION = 2,
    // Do we use half epilogue for the 2nd GEMM (hmma_fp32)
    bool BMM2_FP16_EPILOGUE = true,
    // The output type.
    typename OutputType = typename Traits::A_type,
    // The sage attention block size for Q, K and V
    int SAGE_BLOCK_SIZE_Q = 0, int SAGE_BLOCK_SIZE_K = 0, int SAGE_BLOCK_SIZE_V = 0,
    // Enable skip softmax attention feature.
    bool ENABLE_SKIP_SOFTMAX = false>
using Kernel_traits_v2_paged_kv_cache = Kernel_traits_<Traits, fmha::v2::Gmem_tile_q_k_v, fmha::v2::Gmem_tile_paged_kv,
    fmha::v2::Gmem_tile_paged_kv, fmha::Gmem_tile_o_dispatcher<Traits, OutputType>::template Gmem_tile_o, S, D, DV,
    STEP, WARPS_M, WARPS_N, CTAS_PER_HEAD, FLAGS, 2, MASK_VERSION, BMM2_FP16_EPILOGUE, SAGE_BLOCK_SIZE_Q,
    SAGE_BLOCK_SIZE_K, SAGE_BLOCK_SIZE_V, ENABLE_SKIP_SOFTMAX>;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha_sm120

TRTLLM_MV2_NAMESPACE_END
