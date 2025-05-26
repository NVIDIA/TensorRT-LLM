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

#include <fmha/gemm.h>
#include <fmha/hopper/arrive_wait.h>
#include <fmha/hopper/kernel_traits.h>
#include <fmha/hopper/utils_warpgroup.h>
#include <fused_multihead_attention_kernel.h>

namespace fused_multihead_attention
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void device_4x1_hopper_nl(Params const& params)
{
    // The instruction traits for P.
    using Traits_p = typename Kernel_traits::Traits_p;
    // The instruction traits for O.
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

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    // The shared memory tile to swizzle O.
    // using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    // The compute tile for P.
    using Compute_tile_p = typename Kernel_traits::Compute_tile_p;

    // The compute tile for o.
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    // Do we use LDGSTS for Q, K or V?
    enum
    {
        USE_LDGSTS_Q = Kernel_traits::USE_LDGSTS_Q
    };

    enum
    {
        USE_LDGSTS_K = Kernel_traits::USE_LDGSTS_K
    };

    enum
    {
        USE_LDGSTS_V = Kernel_traits::USE_LDGSTS_V
    };

    // Do we use LDGSTS for any of the 3 input matrices.
    enum
    {
        USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V
    };

    // If either K or V uses LDGSTS, they cannot share a buffer.
    static_assert(!(USE_LDGSTS_K || USE_LDGSTS_V) || !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");

    // Shared memory.
    extern __shared__ char smem_[];

    char* q_smem_ = &smem_[0];
    // It is good to make sure the start address of SMEM is 1024B aligned.
    q_smem_ = fmha::align_1024(q_smem_);
    char* k_smem_ = &q_smem_[Smem_tile_q::BYTES_PER_TILE];
    char* v_smem_ = &k_smem_[Smem_tile_k::BYTES_PER_TILE];
    char* softmax_smem_ = nullptr; // no smem needed

    // we should make sure that SMEM address is 1024B aligned.

    // The loop -- each CTA works on a different loop iteration.
    int const loop = blockIdx.z;
    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
    if (binfo.stop_early(loop))
    {
        return;
    }

    // Create the object to control the masks.
    fmha::Mask_hopper<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx, loop * Gmem_tile_q::ROWS);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(q_smem_, tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(k_smem_, tidx);

    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // Allocate the shared memory tile loader for V.
    Smem_tile_v smem_v(v_smem_, tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx, loop * Gmem_tile_o::ROWS);

    // Trigger the loads for V.
    gmem_v.load(smem_v);
    // If needed, push the LDGDEPBAR instruction after the loads for V.
    fmha::ldgdepbar<USE_LDGSTS && Smem_tile_v::TRANSPOSE>();

    // Trigger the loads for Q at 0th STEP.
    gmem_q.load(smem_q);

    // Trigger the loads for K.
    gmem_k.load(smem_k);

    // Push the LDGDEPBAR instruction after the loads for Q and K.
    fmha::ldgdepbar<USE_LDGSTS>();

    if (Smem_tile_v::TRANSPOSE)
    {
        // Wait for V to be available in SMEM: up to two ldgsts can be outstanding (q0+k above)
        fmha::depbar_<USE_LDGSTS, 1>();
        __syncthreads();
        // For 8-bit data types we have to transpose V in SMEM to be in Column-major.
        smem_v.transpose_tile(tidx);
        // Fence to guarantee ordering between STSM and GMMA.
        fmha::fence_view_async_shared();
        // Not needed as we call it for BMM1.
        // fmha::warpgroup_arrive();
    }

    // Store/load P to/from memory (for debugging).
#if defined(STORE_P)
    enum
    {
        BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8
    };

    using Gmem_tile_p = fmha::Gmem_tile_ps_hopper<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
    char* p_ptr = reinterpret_cast<char*>(params.p_ptr);
    p_ptr += loop * Cta_tile_p::M * params.p_stride_in_bytes;
    Gmem_tile_p gmem_p(p_ptr, params.p_stride_in_bytes, params.scale_bmm1, tidx);
#endif

    // Store S to memory (for debugging). NOTE: We use A_type as C_type is int32 for IMMA???
#if defined(STORE_S)
    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits_p::A_type) * 8
    };

    using Gmem_tile_s = fmha::Gmem_tile_ps_hopper<Traits_p, Cta_tile_p, BITS_PER_ELT_S>;
    char* s_ptr = reinterpret_cast<char*>(params.s_ptr);
    s_ptr += loop * Cta_tile_p::M * params.s_stride_in_bytes;
    Gmem_tile_s gmem_s(s_ptr, params.s_stride_in_bytes, params.scale_softmax, tidx);
#endif

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
    // softmax for hopper should not require SMEM. Maybe pass a nullptr.
    Softmax softmax(params, softmax_smem_, bidb, tidx);

    // Make sure the data is in shared memory.
    fmha::depbar<USE_LDGSTS_Q, 2>();
    __syncthreads(); // At this point, no LDGSTS outstanding (N-2!)
    // GEMM 0.

    // Let's try to use compute_tile for now.
    // Need to think about refactoring into gemm class [Timmy]

    // Compute_tile for P.
    Compute_tile_p compute_tile_p(q_smem_, k_smem_);
    compute_tile_p.clear();

    // for now let's not pipeline GMMA yet.
    // promise to compiler that data are ready in SMEM
    fmha::warpgroup_arrive();
#pragma unroll
    for (int mmas_k_idx = 0; mmas_k_idx < Mma_tile_p::MMAS_K - 1; ++mmas_k_idx)
    {
        compute_tile_p.compute(mmas_k_idx);
    }
    // Last GMMA increments score board.
    compute_tile_p.compute(Mma_tile_p::MMAS_K - 1, true, true);
    // All preceding GMMAs are finished.

    // Load the mask for that iteration.
    mask.load(loop);

    fmha::warpgroup_wait<0>();

    // Softmax.
    // Store the P matrix.
#if defined(STORE_P)
    gmem_p.store(compute_tile_p.acc_);
    gmem_p.move();
#endif

    // Convert from the accumulator type to FP32 for Softmax.
    // Note that alpha is also applied here.
    softmax.unpack(compute_tile_p.acc_);

    // Apply the mask.
    if (params.has_alibi)
    {
        softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
    }
    else
    {
        softmax.apply_mask(mask);
    }

    // Make sure we are done reading the data.
    // For Hopper, most likely it is not shared.
    if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V && loop == 0)
    {
        __syncthreads();
    }
    float p_max[Softmax::ROWS_PER_THREAD * Softmax::MMAS_M];
    // Enable our trick to use the max for INT8 to scale.
    if (Kernel_traits::USE_SCALE_MAX)
    {
        // 16129 == 127 ^ 2.
        // float p_max = reinterpret_cast<const float&>(params.scale_bmm1) * 16129.f;
        // softmax.apply_exp(p_max);
    }
    else
    {
        // Compute the max.
        softmax.template reduce<fmha::Max_>(p_max);

        // Make sure we are done reading shared memory.
        // We don't really use SMEM currently in Hopper for softmax reduction.
        //__syncthreads();

        // Compute the exponential value.
        softmax.apply_exp(p_max);
    }

    // Compute the sum.
    float p_sum[Softmax::ROWS_PER_THREAD * Softmax::MMAS_M];
    softmax.template reduce<fmha::Sum_>(p_sum);

    // Finalize softmax on the accumulators of P^T.
    softmax.scale(p_sum);

    // Store the P matrix.
#if defined(STORE_S)
    softmax.store(gmem_s);
    gmem_s.move();
#endif

    // GEMM 1.

    // compute_tile for o. ( should take care of the 64x64xS tile. )
    Compute_tile_o compute_tile_o(v_smem_, v_smem_);
    compute_tile_o.clear();

    // Repack for the next BMM.
    fmha::Fragment_a<Traits_o, fmha::Row> frag_s[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
    constexpr int NUM_KGROUPS = Smem_tile_v::BUFFERS_PER_TILE;
    constexpr int MMAS_K_PER_GROUP = Mma_tile_o::MMAS_K / NUM_KGROUPS;
    static_assert(MMAS_K_PER_GROUP * NUM_KGROUPS == Mma_tile_o::MMAS_K);

    // Fill frag_s with the results from softmax
    softmax.pack(frag_s);

    // for now let's not pipeline GMMA yet.
    // promise to compiler that data are ready in SMEM
    fmha::warpgroup_arrive();
#pragma unroll
    for (int kbi = 0; kbi < NUM_KGROUPS - 1; kbi++)
    {
#pragma unroll
        for (int ki = 0; ki < MMAS_K_PER_GROUP; ki++)
        {
            compute_tile_o.fill_frag_a(frag_s[kbi * MMAS_K_PER_GROUP + ki][0]);
            // Never increment scoreboard, but check for last kblock.
            compute_tile_o.compute(ki, false, ki == MMAS_K_PER_GROUP - 1);
        }
        compute_tile_o.increment_gmma_desc_group();
    }

#pragma unroll
    for (int ki = 0; ki < MMAS_K_PER_GROUP - 1; ki++)
    {
        compute_tile_o.fill_frag_a(frag_s[(NUM_KGROUPS - 1) * MMAS_K_PER_GROUP + ki][0]);
        compute_tile_o.compute(ki);
    }

    compute_tile_o.fill_frag_a(frag_s[NUM_KGROUPS * MMAS_K_PER_GROUP - 1][0]);
    compute_tile_o.compute(NUM_KGROUPS * MMAS_K_PER_GROUP - 1, true, true);
    // all preceding GMMAs are finished.
    fmha::warpgroup_wait<0>();

#ifdef DEBUG_HAS_PRINT_BUFFER
    using Acc = fmha::Fragment_accumulator<Traits_o>;

    // static_assert(Acc::NUM_REGS == 64 * 64 / 128); // 32
    auto& tmp = compute_tile_o.acc_[0][0];

    float* ptr = reinterpret_cast<float*>(params.print_ptr);
    if (loop == 1 && tidx < 4 && bidb == 0 && bidh == 0)
    {
        float2 reg = fmha::half2_to_float2(tmp.reg(0));
        float x = tmp.elt(0);
        float y = tmp.elt(1);
        ptr[tidx * 2 + 0] = x;
        ptr[tidx * 2 + 1] = y;
    }
#endif

    // store O matrix.
    gmem_o.store(compute_tile_o.acc_);

    if (params.softmax_stats_ptr != nullptr)
    {
        using Mma_tile = typename Traits_p::template Mma_tile<Cta_tile_o>;
        fmha::Softmax_saver<Cta_tile_o, Mma_tile> saver(params, binfo);
        // float scale_bmm1 = Kernel_traits::USE_SCALE_MAX ? reinterpret_cast<const float&>(params.scale_bmm1) :
        // 0.0;//TODO
        saver.store(loop, p_sum, p_max);
    }

} // kernel

} // namespace fused_multihead_attention
