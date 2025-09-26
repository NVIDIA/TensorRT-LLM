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
inline __device__ void device_4xN_hopper(Params const& params)
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
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

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
    char* o_smem_ = &v_smem_[Smem_tile_v::BYTES_PER_TILE];
    char* softmax_smem_ = &o_smem_[Smem_tile_o::BYTES_PER_TILE];

    // we should make sure that SMEM address is 1024B aligned.

    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The block index.
    // const int bidx = bidb * gridDim.x + bidh;
    // The thread index.
    int const tidx = threadIdx.x;

    Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
    if (binfo.stop_early())
    {
        return;
    }

    // Create the object to control the masks.
    fmha::Mask_hopper<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
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
    Gmem_tile_o gmem_o(params, binfo, tidx);

    Smem_tile_o smem_o(o_smem_, tidx);

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

    // Commit the data for Q and K to shared memory.
    // Let's not commit as we always use LDGSTS/TMA for Hopper.
    // gmem_q.commit(smem_q);
    // gmem_k.commit(smem_k);

    smem_q.move_next_write_buffer();
    gmem_q.move();
    // Trigger the loads for Q at 1th STEP
    gmem_q.load(smem_q);
    // Push the LDGDEPBAR instruction after the next load of Q.
    fmha::ldgdepbar<USE_LDGSTS>();

    if (Smem_tile_v::TRANSPOSE)
    {
        // Wait for V to be available in SMEM: up to two ldgsts can be outstanding (q0+k, q1 above)
        fmha::depbar_<USE_LDGSTS, 2>();
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
    Gmem_tile_p gmem_p(params.p_ptr, params.p_stride_in_bytes, params.scale_bmm1, tidx);
#endif

    // Store S to memory (for debugging). NOTE: We use A_type as C_type is int32 for IMMA???
#if defined(STORE_S)
    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits_p::A_type) * 8
    };

    using Gmem_tile_s = fmha::Gmem_tile_ps_hopper<Traits_p, Cta_tile_p, BITS_PER_ELT_S>;
    Gmem_tile_s gmem_s(params.s_ptr, params.s_stride_in_bytes, params.scale_softmax, tidx);
#endif

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
    // softmax for hopper should not require SMEM. Maybe pass a nullptr.
    Softmax softmax(params, softmax_smem_, bidb, tidx);

// Load over the entire sequence length.
// The number of loops should be the number of STEPS
#pragma unroll 1
    for (int loop = 0, outer = 0; loop < Cta_tile_p::N; loop += Cta_tile_p::M, outer++)
    {
        // Make sure the data is in shared memory for this loop.
        fmha::depbar<USE_LDGSTS_Q, 3>();
        __syncthreads(); // At this point, only one LDGSTS outstanding (N-2!)
        // GEMM 0.

        // Let's try to use compute_tile for now.
        // Need to think about refactoring into gemm class [Timmy]

        // q_smem_ address should be updated per STEP.
        // Kind of a hack for now as it assumes 2 buffers for Q.
        char* q_smem_per_step = q_smem_ + (outer % 2) * Smem_tile_q::BYTES_PER_BUFFER;
        // compute_tile for P. ( should take care of the 64x512x64 tile. )
        int warp = tidx / 32;
        int warp_n = warp / 4;
        // static_assert(Traits_p::GMMA_N == 192);
        //  TODO how to set this up? Also GMMA_N % 8 = 0 to line up with XOR?
        Compute_tile_p compute_tile_p(
            q_smem_per_step, k_smem_ + warp_n * Cta_tile_p::K * Traits_p::GMMA_N * sizeof(typename Traits_p::B_type));
        compute_tile_p.clear();

        static_assert(Compute_tile_p::MMAS_N == 1);

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
        mask.load(outer);

        fmha::warpgroup_wait<0>();

        // we double buffer for Q.
        // the SMEM consumed by current loop is issued by ldgsts 2 loops ahead.
        if (loop + Cta_tile_p::M * 2 < Cta_tile_p::N)
        {
            // if there exist a next loop.
            smem_q.move_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
        }
        // Make sure we have the LDGDEPBAR in place.
        fmha::ldgdepbar<USE_LDGSTS_Q>();

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
        float p_max[Softmax::ROWS_PER_THREAD * Softmax::MMAS_M]{0.f};
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

            if (Cta_tile_p::WARPS_N > 1)
            {
                // Inter warp reduction needed.
                __syncthreads();
            }
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
        Compute_tile_o compute_tile_o(nullptr, v_smem_);
        compute_tile_o.clear();

        // Repack for the next BMM.
        using Frag_a = fmha::Fragment_a<Traits_o, fmha::Row>;
        Frag_a frag_s[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];

        static_assert(Frag_a::NUM_ELTS == 16 * Traits_o::GMMA_K / 32);
        // static_assert(Mma_tile_o::MMAS_K == 6);
        static_assert(Mma_tile_o::MMAS_M == 1);
        // static_assert(Compute_tile_o::MMAS_K == 4);

        // Fill frag_s with the results from softmax
        softmax.pack(frag_s);

        // for now let's not pipeline GMMA yet.
        // promise to compiler that data are ready in SMEM
        fmha::warpgroup_arrive();

        compute_tile_o.compute_incta_splitk(frag_s, warp_n);
        // all preceding GMMAs are finished.
        fmha::warpgroup_wait<0>();

// Loop over MMAS_M.
#pragma unroll
        for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
        {

            // Swizzle the elements and do the final reduction.
            smem_o.store(compute_tile_o.acc_, ii);

            // Make sure the data is in shared memory.
            __syncthreads();

            // Load from shared memory.
            uint4 out[Gmem_tile_o::STGS_PER_LOOP];
            smem_o.load(out);

            // Make sure the data was read from shared memory.
            if (ii < Gmem_tile_o::LOOPS - 1)
            {
                __syncthreads();
            }

            // Output the values.
            gmem_o.store(out, ii);
        }

        gmem_o.move();

        if (params.softmax_stats_ptr != nullptr)
        {
            using Mma_tile = typename Traits_p::template Mma_tile<Cta_tile_o>;
            fmha::Softmax_saver<Cta_tile_o, Mma_tile> saver(params, binfo);
            // float scale_bmm1 = Kernel_traits::USE_SCALE_MAX ? reinterpret_cast<const float&>(params.scale_bmm1) :
            // 0.0;//TODO
            saver.store(outer, p_sum, p_max);
        }

#ifdef DEBUG_HAS_PRINT_BUFFER
        int lane = tidx % 32;
        using Acc = fmha::Fragment_accumulator<Traits_o>;
        float* ptr = reinterpret_cast<float*>(params.print_ptr);
        half* ptr_o = reinterpret_cast<half*>(o_smem_);
        if (loop == 0 && warp_n < 2 && lane == 0)
        {
        }
        __syncthreads();
#endif

    } // for loop
} // kernel

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
