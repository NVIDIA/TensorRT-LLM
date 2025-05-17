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
#include <fmha/kernel_traits.h>
#include <fused_multihead_attention_kernel.h>

namespace fused_multihead_attention
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void device_2x2(Params const& params)
{

    // The instruction traits.
    using Traits_p = typename Kernel_traits::Traits_p;
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
        USE_LDGSTS_V = 0
    };

    // Do we use LDGSTS for any of the 3 input matrices.
    enum
    {
        USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V
    };

    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    // Block info to determine if we stop here.
    Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
    if (binfo.stop_early())
    {
        return;
    }

    // The structure to hold the mask.
    fmha::Mask<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[0], tidx);

    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);

    // Push the LDGDEPBAR instruction after the loads for Q, K and V.
    fmha::ldgdepbar<USE_LDGSTS>();

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_k.commit(smem_k);

    // Make sure the data is in shared memory.
    fmha::depbar<USE_LDGSTS, 1>();
    __syncthreads();

    // Load the fragments for Q.
    typename Smem_tile_q::Fragment frag_q[2][Mma_tile_p::MMAS_M];
    smem_q.load(frag_q[0], 0);

    // Load the fragments for K.
    typename Smem_tile_k::Fragment frag_k[2][Mma_tile_p::MMAS_N];
    smem_k.load(frag_k[0], 0);

    // Declare the accumulators for the 1st gemm.
    fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
    fmha::Clear_accumulator<typename Traits_p::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
    for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
    {

        // Trigger the load from shared memory for the next series of Q/K values.
        smem_q.load(frag_q[ki & 1], ki);
        smem_k.load(frag_k[ki & 1], ki);

        // Do the math for the values already in registers.
        if (ki <= Mma_tile_p::VALID_MMAS_K)
        {
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
    }

    // Do the final stage of math.
    if (Mma_tile_p::MMAS_K <= Mma_tile_p::VALID_MMAS_K)
    {
        int ki = Mma_tile_p::MMAS_K;
        fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
    }

    // Store the P matrix.
#if defined(STORE_P)
    enum
    {
        BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8
    };

    using Gmem_tile_p = fmha::Gmem_tile_ps<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
    Gmem_tile_p gmem_p(params.p_ptr, params.p_stride_in_bytes, params.scale_bmm1, tidx);
    gmem_p.store(acc_p);
#endif

    // Make sure the shared memory was consumed.
    __syncthreads();

    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_, tidx);
    // Trigger the loads for V.
    gmem_v.load(smem_v);

    // Load the mask.
    mask.load(0);

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, &smem_[0], bidb, tidx);

    // Convert from the accumulator type to FP32 for Softmax.
    softmax.unpack(acc_p);

    // Apply the mask.
    if (params.has_alibi)
    {
        softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
    }
    else
    {
        softmax.apply_mask(mask);
    }

    // Enable our trick to use the max for INT8 to scale.
    if (params.use_int8_scale_max)
    {
        // 16129 == 127 ^ 2.
        float p_max = reinterpret_cast<float const&>(params.scale_bmm1) * 16129.f;
        softmax.apply_exp(p_max);
    }
    else
    {
        float p_max[Softmax::ROWS_PER_THREAD];
        softmax.template reduce<fmha::Max_>(p_max);

        // Compute the exponential value.
        softmax.apply_exp(p_max);
    }

    // Compute the sum.
    float p_sum[Softmax::ROWS_PER_THREAD];
    softmax.template reduce<fmha::Sum_>(p_sum);

    // Commit the data to shared memory for V. It must happen after the last "reduce".
    gmem_v.commit(smem_v);

    // Make sure the data for V is in shared memory.
    __syncthreads();

    // Finalize softmax on the accumulators of P^T.
    softmax.scale(p_sum);

    // Store the P matrix.
#if defined(STORE_S)
    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits_p::A_type) * 8
    };

    using Gmem_tile_s = fmha::Gmem_tile_ps<Traits_p, Cta_tile_p, BITS_PER_ELT_S>;
    Gmem_tile_s gmem_s(params.s_ptr, params.s_stride_in_bytes, params.scale_softmax, tidx);
    softmax.store(gmem_s);
#endif

    // Repack the transformed P elements to fragments for the next GEMM.
    fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
    softmax.pack(frag_p);

    // Declare the accumulators for the 1st gemm.
    fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
    fmha::Clear_accumulator<typename Traits_o::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[2][Mma_tile_o::VALID_MMAS_N];
    smem_v.load(frag_v[0], 0);

// Do this part of O = P^T * V^T.
#pragma unroll
    for (int ki = 1; ki < Mma_tile_o::MMAS_K; ++ki)
    {
        // Trigger the load from shared memory for the next series of Q/K values.
        smem_v.load(frag_v[ki & 1], ki);
        // Do the math.
        fmha::gemm(acc_o, frag_p[(ki - 1)], frag_v[(ki - 1) & 1]);
    }

    // Do the final stage of math.
    {
        int ki = Mma_tile_o::MMAS_K;
        fmha::gemm(acc_o, frag_p[(ki - 1)], frag_v[(ki - 1) & 1]);
    }

// // DEBUG.
// printf("tidx=%3d acc_o[0][0]=0x%08x\n", tidx, acc_o[0][0].reg(0));
// printf("tidx=%3d acc_o[1][0]=0x%08x\n", tidx, acc_o[1][0].reg(0));
// printf("tidx=%3d acc_o[2][0]=0x%08x\n", tidx, acc_o[2][0].reg(0));
// printf("tidx=%3d acc_o[3][0]=0x%08x\n", tidx, acc_o[3][0].reg(0));
// // END OF DEBUG.

// Loop over MMAS_M.
#pragma unroll
    for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
    {

        // Make sure the data was read from shared memory.
        __syncthreads();

        // Swizzle the elements and do the final reduction.
        smem_o.store(acc_o, ii);

        // Make sure the data is in shared memory.
        __syncthreads();

        // Load from shared memory.
        static_assert(Gmem_tile_o::STGS_PER_LOOP == Smem_tile_o::LDS_PER_LOOP, "");
        uint4 out[Gmem_tile_o::STGS_PER_LOOP];
        smem_o.load(out);

        // // DEBUG.
        // #pragma unroll
        // for( int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; ++jj ) {
        //     printf("tidx=%3d loop=%d out[%d]=0x%08x\n", tidx, ii, jj, out[jj].x);
        // }
        // // END OF DEBUG.

        // Output the values.
        gmem_o.store(out, ii);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
