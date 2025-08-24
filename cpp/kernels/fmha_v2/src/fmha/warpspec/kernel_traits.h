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
#include <cuda/std/array>
#include <fmha/hopper/compute_tile.h>
#include <fmha/hopper/gmem_tile_o_packed.h>
#include <fmha/hopper/smem_tile.h>
#include <fmha/numeric_types.h>
#include <fmha/utils.h>
#include <fmha/warpspec/circular_buffer.h>
#include <fmha/warpspec/epilogue.h>

namespace fmha
{
namespace ws
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction trait template for initializing BMM1 and BMM2 traits.
    template <int, int, int, bool, bool> class Instruction_traits,
    // The step size in query sequence dimension (M of BMM1 and BMM2).
    int STEP_Q_,
    // The step size in key/value sequence dimension (N of BMM1 and K of BMM2).
    int STEP_KV_,
    // The head dimension.
    int D_,
    // The head dimension of V.
    int DV_,
    // The number of smem buffers for Q tiles.
    int Q_BUFFERS_,
    // The number of smem buffers for K, and V tiles.
    int KV_BUFFERS_,
    // The number of compute warpgroups (128 threads per warpgroup).
    int NUM_COMPUTE_GROUPS_,
    // The number of data warpgroups (TMA).
    int DMA2COMPUTE_DEPTH_,
    // The attention mask type: padding (0), causal (1), sliding_window_causal (2), custom_mask (3).
    // See fused_multihead_attention_kernel.h for description.
    int ATTENTION_MASK_TYPE_ = 0,
    // Is head interleaved ?
    // (head_interleaved means input [bxs, h, 3, d], otherwise [bx3, 3, h, d]).
    bool HEADS_INTERLEAVED_ = true,
    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    bool APPLY_ALIBI_ = false,
    // Enable mutex to overlap mma and softmax ?
    bool ENABLE_MUTEX_ = true,
    // The tile scheduling mode: static (0), dynamic unbalanced (1), dynamic balanced (2)
    int SCHEDULING_MODE_ = 0,
    // The qkv input layout: packed_qkv (0), contiguous_q_kv (1), q_paged_kv (2).
    int INPUT_LAYOUT_ = 0,
    // Whether use UTMASTG in epilogue. This is ignored for FP16/BF16 at the moment.
    bool USE_TMA_STORE_ = false,
    // Enable softcapping_scale to the qk products ? (from Grok models)
    bool ENABLE_BMM1_SOFTCAPPING_SCALE_ = false,
    // Save softmax stats ?
    bool RETURN_SOFTMAX_STATS_ = false,
    // The output type (only used by fp8 kernels).
    typename OutputType = typename Instruction_traits<STEP_Q_, STEP_KV_, 0, false, false>::A_type,
    // The sage attention block size for Q, K and V
    int SAGE_BLOCK_SIZE_Q_ = 0, int SAGE_BLOCK_SIZE_K_ = 0, int SAGE_BLOCK_SIZE_V_ = 0>
struct Kernel_traits
{

    // The step size in query sequence dimension (M of BMM1 and BMM2).
    enum
    {
        STEP_Q = STEP_Q_
    };

    // The step size in key/value sequence dimension (N of BMM1 and K of BMM2).
    enum
    {
        STEP_KV = STEP_KV_
    };

    // The valid head dimension.
    enum
    {
        VALID_D = D_
    };

    enum
    {
        VALID_DV = (DV_ == 0 ? D_ : DV_)
    };

    // Bootstrap GMMA_K from dummy Instruction_traits where FP16/BF16 K = 16, FP8 K = 32.
    enum
    {
        GMMA_K = Instruction_traits<STEP_Q, STEP_KV, 0, false, false>::GMMA_K
    };

    // The instruction traits for the BMM1.
    using Traits_p = Instruction_traits<STEP_Q, STEP_KV, GMMA_K, false, false>;

    // The element type.
    using Element_data_type = typename Traits_p::A_type;

    // The bytes per element.
    enum
    {
        ELEMENT_BYTES = sizeof(Element_data_type)
    };

    // The padded head dimension.
    enum
    {
        D = std::min<int>(Round_up<VALID_D, 128 / ELEMENT_BYTES>::VALUE, Next_power_of_two<VALID_D>::VALUE)
    };

    enum
    {
        DV = std::min<int>(Round_up<VALID_DV, 128 / ELEMENT_BYTES>::VALUE, Next_power_of_two<VALID_DV>::VALUE)
    };

    // The number of smem buffers for Q tiles.
    enum
    {
        Q_BUFFERS = Q_BUFFERS_
    };

    // The number of smem buffers for K, and V tiles.
    enum
    {
        KV_BUFFERS = KV_BUFFERS_
    };

    // Whether read from paged kv buffers or not.
    enum
    {
        PAGED_KV_INPUT = INPUT_LAYOUT_ == 2
    };

    // Whether Q and KV is in separate buffer, which means we need to consider different Q and KV lengths.
    enum
    {
        SEPARATE_Q_KV_BUFFER = INPUT_LAYOUT_ > 0
    };

    // Whether use UTMASTG in epilogue. This is always false for FP16/BF16 at the moment.
    enum
    {
        USE_TMA_STORE = 0
    };

    // SageAttention needs fp8 input
    enum
    {
        SAGE_ATTENTION = SAGE_BLOCK_SIZE_Q_ > 0 || SAGE_BLOCK_SIZE_K_ > 0 || SAGE_BLOCK_SIZE_V_ > 0
    };

    enum
    {
        SAGE_BLOCK_SIZE_Q = SAGE_BLOCK_SIZE_Q_
    };

    enum
    {
        SAGE_BLOCK_SIZE_K = SAGE_BLOCK_SIZE_K_
    };

    enum
    {
        SAGE_BLOCK_SIZE_V = SAGE_BLOCK_SIZE_V_
    };

    // Whether the dma group transposes the v tile explicitly.
    enum
    {
        DMA_GROUP_TRANSPOSE_V
        = (std::is_same<Element_data_type, fmha::e4m3_t>::value || std::is_same<Element_data_type, fmha::e5m2_t>::value)
    };

    // The number of smem scratch buffer for staging V transpose for Hopper QGMMA
    enum
    {
        V_SCRATCH_BUFFERS = DMA_GROUP_TRANSPOSE_V ? 1 : 0
    };

    // The number of compute warpgroups (128 threads per warpgroup).
    enum
    {
        NUM_COMPUTE_GROUPS = NUM_COMPUTE_GROUPS_
    };

    // The number of data warpgroups (TMA).
    enum
    {
        DMA2COMPUTE_DEPTH = DMA2COMPUTE_DEPTH_
    };

    // The number of ctas per cluster.
    enum
    {
        CTAS_PER_CGA = 1
    };

    // The total number of threads per block,
    enum
    {
        THREADS = 128 + NUM_COMPUTE_GROUPS * 128
    };

    // The number of warps in the M dimension.
    enum
    {
        WARPS_M = 4
    };

    // The number of warpgroups in the M dimensions.
    enum
    {
        WARP_GROUP_M = WARPS_M / 4
    };

    // The number of warps in the N dimension.
    enum
    {
        WARPS_N = 1
    };

    // The number of warpgroups in the N dimension.
    enum
    {
        WARP_GROUP_N = WARPS_N
    };

    // The number of warpgroups in the K dimension.
    enum
    {
        WARP_GROUP_K = 1
    };

    // The attention mask type: padding (0), causal (1), sliding_or_chunked_attention (2), custom_mask (3).
    enum
    {
        CAUSAL_MASK = (ATTENTION_MASK_TYPE_ == 1 || ATTENTION_MASK_TYPE_ == 2)
    };

    enum
    {
        SLIDING_OR_CHUNKED_ATTENTION = ATTENTION_MASK_TYPE_ == 2
    };

    // Is head interleaved ?
    // (head_interleaved means input [bxs, h, 3, d], otherwise [bx3, 3, h, d]).
    enum
    {
        HEADS_INTERLEAVED = HEADS_INTERLEAVED_
    };

    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    enum
    {
        APPLY_ALIBI = APPLY_ALIBI_
    };

    // Are we save the softmax stats?
    enum
    {
        RETURN_SOFTMAX_STATS = RETURN_SOFTMAX_STATS_
    };

    // Are we applying softcapping scale for qk products ?
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = ENABLE_BMM1_SOFTCAPPING_SCALE_
    };

    // Use the custom mask input ( attention_mask_type == 3.)
    enum
    {
        USE_CUSTOM_MASK = ATTENTION_MASK_TYPE_ == 3
    };

    static_assert(!USE_CUSTOM_MASK || STEP_KV == 64 || STEP_KV == 128 || STEP_KV == 256, "Not implemented!");

    // Apply the exp2f optimization (fuse bmm1_scale and -max into FMAs).
    // Performance degradation when enabled exp2f tricks with dense mask.
    // with softcapping scale, exp2f optimization cannot work.
    enum
    {
        EXP2F_OPTIMIZATION = !APPLY_ALIBI && !ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Enable mutex to overlap mma and softmax ?
    enum
    {
        ENABLE_MUTEX = ENABLE_MUTEX_
    };

    // The tile scheduling mode: static (0), dynamic (1)
    enum
    {
        SCHEDULING_MODE = SCHEDULING_MODE_
    };

    // The bytes of head dimension.
    enum
    {
        D_BYTES = D * ELEMENT_BYTES
    };

    // Split D into multiple groups in order to match the TMA swizzle mode (128B).
    // 1. BMM1: we split D into multiple K groups.
    // 2. BMM2: we split D into multiple N groups,
    //          but only have one MMA_N as we can use leading_dim_offset to handle this.

    // The number of head_dimension groups.
    enum
    {
        D_GROUPS = fmha::Div_up<D_BYTES, 128>::VALUE
    };

    // The head_dimension per group.
    enum
    {
        D_PER_GROUP = D / D_GROUPS
    };

    static_assert(D_GROUPS * D_PER_GROUP == D);

    // The head_dimension bytes per group
    enum
    {
        D_BYTES_PER_GROUP = D_BYTES / D_GROUPS
    };

    // The bytes of head dimension of V.
    enum
    {
        DV_BYTES = DV * ELEMENT_BYTES
    };

    // The number of head_dimension groups of V.
    enum
    {
        DV_GROUPS = fmha::Div_up<DV_BYTES, 128>::VALUE
    };

    // QGMMA: BMM2 will be split into multiple K groups as we explicitly transpose v (128 * D) in the smem.
    // HGMMA: BMM2 will load from row-major (K * N) smem_v, so we don't need to explicitly split K.
    static constexpr auto BMM2_LEADING_DIM_BYTES = ELEMENT_BYTES == 1 ? 128 : STEP_KV * ELEMENT_BYTES;

    enum
    {
        BMM2_K_GROUPS = fmha::Div_up<STEP_KV * ELEMENT_BYTES, BMM2_LEADING_DIM_BYTES>::VALUE
    };

    // The K dimension per group
    enum
    {
        BMM2_K_PER_GROUP = fmha::Div_up<STEP_KV, BMM2_K_GROUPS>::VALUE
    };

    // The K dimension bytes per group
    enum
    {
        BMM2_K_BYTES_PER_GROUP = ELEMENT_BYTES * BMM2_K_PER_GROUP * BMM2_K_GROUPS
    };

    // Set GMMA descriptor mode based on the head_size.
    static constexpr auto GMMA_DESC_MODE = (D_BYTES_PER_GROUP > 64 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
            : D_BYTES_PER_GROUP > 32                               ? fmha::Gmma_descriptor_mode::SWIZZLE_64B
                                                                   : fmha::Gmma_descriptor_mode::SWIZZLE_32B);

    // Named barrier ids
    static constexpr int DMA_SYNC_BARRIER_ID = 0x1;
    static constexpr int MMA_SYNC_BARRIER_ID = 0x2;

    // How many threads get involved in the dma group.
    enum
    {
        NUM_THREADS_IN_DMA_GROUP = DMA_GROUP_TRANSPOSE_V ? 128 : (PAGED_KV_INPUT ? 1 : 32)
    };

    // The instruction traits for the BMM2.
    // FP16/BF16 K = 16, FP8 K = 32.
    using Traits_o = Instruction_traits<STEP_Q, DV, GMMA_K, true, false>;

    // The CTA description for BMM1.
    using Cta_tile_p =
        typename Traits_p::template Cta_tile<STEP_Q, STEP_KV, D, WARP_GROUP_M, WARP_GROUP_N, WARP_GROUP_K>;

    // The CTA description for BMM1 (after head_dimension is split).
    using Cta_tile_p_split_d =
        typename Traits_p::template Cta_tile<STEP_Q, STEP_KV, D_PER_GROUP, WARP_GROUP_M, WARP_GROUP_N, WARP_GROUP_K>;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Traits_o::template Cta_padded_tile<STEP_Q, DV, STEP_KV, VALID_DV, STEP_KV, WARP_GROUP_M,
        WARP_GROUP_K, WARP_GROUP_N>;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

    // Smem_tiles are currently only used as meta data for the compute tile.
    // The Q shared memory tile.
    using Smem_tile_q
        = fmha::Smem_tile_hopper_a<Traits_p, Cta_tile_p_split_d, fmha::Row, 16, Q_BUFFERS * D_GROUPS, GMMA_DESC_MODE,
            true, // USE_TMA_Q
            Traits_p::GMMA_A_RF>;

    // The K shared memory tile.
    using Smem_tile_k
        = fmha::Smem_tile_hopper_b<Traits_p, Cta_tile_p_split_d, fmha::Col, 16, KV_BUFFERS * D_GROUPS, GMMA_DESC_MODE,
            true // USE_TMA_K
            >;

    // The V shared memory tile.
    using Smem_tile_v = fmha::Smem_tile_hopper_b<Traits_o, Cta_tile_o, fmha::Row, 16, KV_BUFFERS, GMMA_DESC_MODE,
        true // USE_TMA_V
        >;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = fmha::Compute_tile_with_gmma<Traits_p, Cta_tile_p, Smem_tile_q, Smem_tile_k,
        Traits_p::GMMA_A_RF, Traits_p::GMMA_B_RF>;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = fmha::Compute_tile_with_gmma<Traits_o, Cta_tile_o, Smem_tile_q, Smem_tile_v,
        Traits_o::GMMA_A_RF, Traits_o::GMMA_B_RF>;

    // The global memory tile for O.
    using Gmem_tile_o = fmha::v2::Gmem_tile_o_hopper<Traits_o, Cta_tile_o, Cta_tile_o::WARPS_K>;

    // The q, k, v tile buffer.
    using Buffer_q_t = cuda::std::array<Element_data_type, D * STEP_Q * Q_BUFFERS>;
    using Buffer_k_t = cuda::std::array<Element_data_type, D * STEP_KV * KV_BUFFERS>;
    using Buffer_v_t = cuda::std::array<Element_data_type, DV * STEP_KV * KV_BUFFERS>;
    // We need one kv buffer to explicitly transose fp8 smem_tile.
    using Buffer_v_scratch_t = cuda::std::array<Element_data_type, DV * STEP_KV * V_SCRATCH_BUFFERS>;

    // The smem bytes of q, k, v tiles.
    enum
    {
        SMEM_BYTES_Q = sizeof(Buffer_q_t),
        SMEM_BYTES_K = sizeof(Buffer_k_t),
        SMEM_BYTES_V = sizeof(Buffer_v_t),
    };

    // The reader/writer (consumer/producer) barriers.
    using Circular_buffer_kv_reader = typename CircularBuffer<KV_BUFFERS, CTAS_PER_CGA>::Reader;
    using Circular_buffer_kv_writer = typename CircularBuffer<KV_BUFFERS, CTAS_PER_CGA>::Writer;
    using Circular_buffer_q_reader = typename CircularBuffer<Q_BUFFERS, CTAS_PER_CGA>::Reader;
    using Circular_buffer_q_writer = typename CircularBuffer<Q_BUFFERS, CTAS_PER_CGA>::Writer;
    using Circular_buffer_v_scratch_reader = typename CircularBuffer<V_SCRATCH_BUFFERS, CTAS_PER_CGA>::Reader;
    using Circular_buffer_v_scratch_writer = typename CircularBuffer<V_SCRATCH_BUFFERS, CTAS_PER_CGA>::Writer;

    // The struct of shared memory buffers.
    struct __align__(128) Shared
    {

        // The smem buffer of q, k, v tiles
        Buffer_q_t smem_q[NUM_COMPUTE_GROUPS];
        Buffer_k_t smem_k;
        Buffer_v_t smem_v;
        uint32_t tile_id;

        // The head info to be shared among compute groups
        struct Head_info
        {
            // How many steps to execute.
            int q_steps;
            // The start tile offset for query.
            int local_q_tile_offset;

            union
            {
                // The start tile offset for query (counting the past query length).
                // Used by fixed-pattern mask types like padding, causal, sliding_window_causal
                int q_tile_offset;
                // The mask sum_s.
                // Used by custom mask input.
                int mask_sum_s;
            };

            // How many steps to execute.
            int kv_steps;
            // The actual query sequence length (variable sequence length).
            int actual_seqlen;
            // The actual key/value sequence length (variable sequence length).
            int actual_kv_seqlen;
            // The batch/head index.
            int bidx;
            // The head index.
            int bidh;
            int bidb;
        };

        // DMA to Compute:
        // In this use case it probably makes sense to have the same number of BUFFERS in both queues.
        // - barriers to wait for K+V loads to complete.
        CircularBuffer<KV_BUFFERS, CTAS_PER_CGA> tma_k_tracker;
        CircularBuffer<KV_BUFFERS, CTAS_PER_CGA> tma_v_tracker;
        CircularBuffer<Q_BUFFERS, CTAS_PER_CGA> tma_q_tracker[NUM_COMPUTE_GROUPS];
        // Not used for fp16/bf16 kernels.
        CircularBuffer<V_SCRATCH_BUFFERS, CTAS_PER_CGA> tma_v_scratch_tracker;
        CircularBufferWithData<DMA2COMPUTE_DEPTH, Head_info> head_info_tracker[NUM_COMPUTE_GROUPS];

        // Mutex
        OrderedMutex compute_mutex;

        inline __device__ void init(int tid0)
        {

#pragma unroll
            for (int i = 0; i < NUM_COMPUTE_GROUPS; i++)
            {
                tma_q_tracker[i].init(tid0, 1, CTAS_PER_CGA);
                head_info_tracker[i].init(tid0, /*producer_threads=*/1, /*consumer_threads=*/128);
            }

            tma_k_tracker.init(tid0, 1, NUM_COMPUTE_GROUPS);
            tma_v_tracker.init(tid0, 1, NUM_COMPUTE_GROUPS);

            compute_mutex.init(tid0, 128, 128);
        }
    };

    enum
    {
        BYTES_PER_SMEM = sizeof(Shared)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Specialized kernel traits for Hopper_qgmma_e4m3_fp32_traits.
template < // The step size in query sequence dimension (M of BMM1 and BMM2).
    int STEP_Q_,
    // The step size in key/value sequence dimension (N of BMM1 and K of BMM2).
    int STEP_KV_,
    // The head dimension.
    int D_,
    // The head dimension of V.
    int DV_,
    // The number of smem buffers for Q tiles.
    int Q_BUFFERS_,
    // The number of smem buffers for K, and V tiles.
    int KV_BUFFERS_,
    // The number of compute warpgroups (128 threads per warpgroup).
    int NUM_COMPUTE_GROUPS_,
    // The number of data warpgroups (TMA).
    int DMA2COMPUTE_DEPTH_,
    // The attention mask type: padding (0), causal (1), sliding_window_causal (2).
    // See fused_multihead_attention_kernel.h for description.
    int ATTENTION_MASK_TYPE_ = 0,
    // Is head interleaved ?
    // (head_interleaved means input [bxs, h, 3, d], otherwise [bx3, 3, h, d]).
    bool HEADS_INTERLEAVED_ = true,
    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    bool APPLY_ALIBI_ = false,
    // Enable mutex to overlap mma and softmax ?
    bool ENABLE_MUTEX_ = true,
    // The tile scheduling mode: static (0), dynamic unbalanced (1), dynamic balanced (2).
    int SCHEDULING_MODE_ = 0,
    // The qkv input layout: packed_qkv (0), contiguous_q_kv (1), q_paged_kv (2).
    int INPUT_LAYOUT_ = 0,
    // Whether use UTMASTG in epilogue. This is ignored for FP16/BF16 at the moment.
    bool USE_TMA_STORE_ = false,
    // Enable softcapping scale to the qk products ? (from Grok models)
    bool ENABLE_BMM1_SOFTCAPPING_SCALE_ = false,
    // Save softmax stats ?
    bool RETURN_SOFTMAX_STATS_ = false,
    // The output type (only used by fp8 kernels).
    typename OutputType = e4m3_t,
    // The sage attention block size for Q, K and V
    int SAGE_BLOCK_SIZE_Q_ = 0, int SAGE_BLOCK_SIZE_K_ = 0, int SAGE_BLOCK_SIZE_V_ = 0>
struct Kernel_traits_Hopper_qgmma_e4m3_fp32
    : public Kernel_traits<Hopper_qgmma_e4m3_fp32_traits, STEP_Q_, STEP_KV_, D_, DV_, Q_BUFFERS_, KV_BUFFERS_,
          NUM_COMPUTE_GROUPS_, DMA2COMPUTE_DEPTH_, ATTENTION_MASK_TYPE_, HEADS_INTERLEAVED_, APPLY_ALIBI_,
          ENABLE_MUTEX_, SCHEDULING_MODE_, INPUT_LAYOUT_, USE_TMA_STORE_, ENABLE_BMM1_SOFTCAPPING_SCALE_,
          RETURN_SOFTMAX_STATS_, OutputType, SAGE_BLOCK_SIZE_Q_, SAGE_BLOCK_SIZE_K_, SAGE_BLOCK_SIZE_V_>
{

    // Base class.
    using Base = Kernel_traits<Hopper_qgmma_e4m3_fp32_traits, STEP_Q_, STEP_KV_, D_, DV_, Q_BUFFERS_, KV_BUFFERS_,
        NUM_COMPUTE_GROUPS_, DMA2COMPUTE_DEPTH_, ATTENTION_MASK_TYPE_, HEADS_INTERLEAVED_, APPLY_ALIBI_, ENABLE_MUTEX_,
        SCHEDULING_MODE_, INPUT_LAYOUT_, USE_TMA_STORE_, ENABLE_BMM1_SOFTCAPPING_SCALE_, RETURN_SOFTMAX_STATS_,
        OutputType, SAGE_BLOCK_SIZE_Q_, SAGE_BLOCK_SIZE_K_, SAGE_BLOCK_SIZE_V_>;

    enum
    {
        USE_TMA_STORE = USE_TMA_STORE_
    };

    enum
    {
        O_BUFFERS = USE_TMA_STORE ? 1 : 0
    };

    // Inherit Traits_o, Cta_tile_o, Smem_tile_q.
    using Traits_o = typename Base::Traits_o;
    using Cta_tile_o = typename Base::Cta_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;

    // The V shared memory tile.
    // For true case below, as QGMMA only supports K-major, Smem_tile_v remaps row-major to col-major
    // as well as the GMMA descriptor for V.
    using Smem_tile_v = fmha::Smem_tile_v<Traits_o, Cta_tile_o, Base::KV_BUFFERS, Base::GMMA_DESC_MODE, false>;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = fmha::Compute_tile_with_gmma<Traits_o, Cta_tile_o, Smem_tile_q, Smem_tile_v,
        Traits_o::GMMA_A_RF, Traits_o::GMMA_B_RF>;

    // The global memory tile for O.
    using Gmem_tile_o = std::conditional_t<std::is_same_v<OutputType, e4m3_t>,
        // e4m3 output
        fmha::v2::Gmem_tile_o_hopper_32bit_8bit<Traits_o, Cta_tile_o, Cta_tile_o::WARPS_K, USE_TMA_STORE>,
        // fp16/bf16 output
        fmha::v2::Gmem_tile_o_qgmma_fp32_16bits<Traits_o, Cta_tile_o, OutputType>>;

    // Inherit Buffer qkv class.
    using Buffer_q_t = typename Base::Buffer_q_t;
    using Buffer_k_t = typename Base::Buffer_k_t;
    using Buffer_v_t = typename Base::Buffer_v_t;
    // We need one kv buffer to explicitly transose fp8 smem_tile.
    using Buffer_v_scratch_t = typename Base::Buffer_v_scratch_t;
    // Extra O buffer if TMA is used for epilogue
    using Element_data_type = typename Base::Element_data_type;
    using Buffer_o_t = cuda::std::array<Element_data_type, Base::DV * Base::STEP_Q * O_BUFFERS>;

    // The struct of shared memory buffers.
    struct __align__(128) Shared
    {

        // The smem buffer of q, k, v tiles
        Buffer_q_t smem_q[Base::NUM_COMPUTE_GROUPS];
        Buffer_k_t smem_k;
        Buffer_v_t smem_v;
        Buffer_v_scratch_t smem_v_scratch;
        Buffer_o_t smem_o[Base::NUM_COMPUTE_GROUPS];
        uint32_t tile_id;

        // The head info to be shared among compute groups
        struct Head_info
        {
            // How many steps to execute.
            int q_steps;
            // The start tile offset for query.
            int local_q_tile_offset;

            union
            {
                // The start tile offset for query (counting the past query length).
                // Used by fixed-pattern mask types like padding, causal, sliding_window_causal
                int q_tile_offset;
                // The mask sum_s.
                // Used by custom mask input.
                int mask_sum_s;
            };

            // How many steps to execute.
            int kv_steps;
            // The actual query sequence length (variable sequence length).
            int actual_seqlen;
            // The actual key/value sequence length (variable sequence length).
            int actual_kv_seqlen;
            // The batch/head index.
            int bidx;
            // The head index.
            int bidh;
            int bidb;
        };

        // DMA to Compute:
        // In this use case it probably makes sense to have the same number of BUFFERS in both queues.
        // - barriers to wait for K+V loads to complete.
        CircularBuffer<Base::KV_BUFFERS, Base::CTAS_PER_CGA> tma_k_tracker;
        CircularBuffer<Base::KV_BUFFERS, Base::CTAS_PER_CGA> tma_v_tracker;
        CircularBuffer<Base::Q_BUFFERS, Base::CTAS_PER_CGA> tma_q_tracker[Base::NUM_COMPUTE_GROUPS];
        CircularBuffer<Base::V_SCRATCH_BUFFERS, Base::CTAS_PER_CGA> tma_v_scratch_tracker;
        CircularBufferWithData<Base::DMA2COMPUTE_DEPTH, Head_info> head_info_tracker[Base::NUM_COMPUTE_GROUPS];

        // Mutex
        OrderedMutex compute_mutex;

        inline __device__ void init(int tid0)
        {

#pragma unroll
            for (int i = 0; i < Base::NUM_COMPUTE_GROUPS; i++)
            {
                tma_q_tracker[i].init(tid0, 1, Base::CTAS_PER_CGA);
                head_info_tracker[i].init(tid0, /*producer_threads=*/1, /*consumer_threads=*/128);
            }

            tma_k_tracker.init(tid0, 1, Base::NUM_COMPUTE_GROUPS);
            tma_v_tracker.init(tid0, 1, Base::NUM_COMPUTE_GROUPS);

            tma_v_scratch_tracker.init(tid0, 1, 1); // producer/consumer in same warp

            compute_mutex.init(tid0, 128, 128);
        }
    };

    enum
    {
        BYTES_PER_SMEM = sizeof(Shared)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws
} // namespace fmha
