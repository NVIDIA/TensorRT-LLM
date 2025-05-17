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

#include "fused_multihead_attention_fprop_kernel.h"
#include <fmha/gemm.h>
#include <fmha/kernel_traits.h>

namespace fused_multihead_attention
{

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Shared Memory Requirements:
 * Q:        2KB
 * K:       32KB
 * V:       32KB
 * D:        4KB
 * S:        8KB
 * O:       16KB
 * Softmax: 256B
 * TOTAL:
 *
 * In this kernel we store D in the sign bit of S, and we do not swizzle the S matrix, since we want
 * to load it in the exact same register layout in the dgrad kernel.
 *
 * Shared Memory Layout:
 * |0 |2                           |18K           |26K  |30K   |34K
 * ------------------------------------------------------------------
 * |Q |                          K                             |
 * |Q |                          V                             |
 * |Q |            O               | Softmax
 *
 *
 * If we want to perform the epilogue swizzle of S and D, the layout would be
 * |0 |2                           |18K           |26K  |30K   |34K
 * |Q |            O               |       S      |  D  | Softmax
 *
 */

template <typename Kernel_traits, bool IS_TRAINING, typename Params, typename Prng>
inline __device__ void device_1xN_(
    Params const& params, int const bidb, int const bidh, int const begin, int const steps, Prng& ph)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = typename Traits::template Mma_tile<Cta_tile_o>;

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

    using Gmem_tile_s = Gmem_tile_mma_s<Traits, Cta_tile_p>;

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

    // The thread index.
    int const tidx = threadIdx.x;

    Block_info_padded<Kernel_traits::THREADS> const binfo(params, bidb, bidh, tidx);
    if (binfo.stop_early())
        return;

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    Gmem_tile_s gmem_s(params, binfo, tidx);
    for (int it = 0; it < begin; it++)
    {
        gmem_q.move();
        gmem_s.move();
        gmem_o.move();
    }

    fmha::Mask<Traits, Cta_tile_p, Kernel_traits::VERSION> mask(params, binfo, tidx);

    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // The base pointer of smem_v;
    char* smem_v_ = nullptr;
    if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
    {
        smem_v_ = &smem_[Smem_tile_q::BYTES_PER_TILE];
    }
    else
    {
        smem_v_ = &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE];
    }
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], tidx);

    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);
    // Trigger the loads for K.
    gmem_v.load(smem_v);

    // Push the LDGDEPBAR instruction after the loads for Q, K and V.
    fmha::ldgdepbar<USE_LDGSTS>();

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_v.commit(smem_v);

    // Commit the data for V to shared memory.
    if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
    {
        gmem_k.commit(smem_k);
    }

    // Make sure the data is in shared memory.
    fmha::depbar<USE_LDGSTS, 1>();
    __syncthreads();

    // Load the fragments for Q.
    typename Smem_tile_q::Fragment frag_q[2][Mma_tile_p::MMAS_M];
    smem_q.load(frag_q[0], 0);

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
#pragma unroll
    for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
    {
        smem_v.load(frag_v[ki], ki);
    }

    // Commit the data for V to shared memory if it has not been done already.
    if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
    {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_k.commit(smem_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }
    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[2][Mma_tile_p::MMAS_N];
    smem_k.load(frag_k[0], 0);

    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits::A_type) * 8
    };

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params,
        &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE], bidb, tidx);

    // The number of threads per row.
    enum
    {
        THREADS_PER_ROW = 32
    };

    // Load over the entire sequence length.
    // for( int loop = 0, outer = 0; loop < Cta_tile_p::N; loop += Cta_tile_p::M, outer++ ) {
    for (int l = 0; l < steps; l++)
    {
        // const int loop = (l + begin) * Cta_tile_p::M;
        // if( loop >= binfo.actual_seqlen ) break;

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
        {

            // Trigger the load from shared memory for the next series of Q values.
            smem_q.load(frag_q[ki & 1], ki);
            smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }

        // Trigger the load for the next Q values.
        // if( loop + Cta_tile_p::M < Cta_tile_p::N ) {
        if (l < steps - 1)
        {
            smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
        }

        // Load the mask for that iteration.
        mask.load(begin + l);

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

        if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0)
        {
            // if we share K and V, it could be that V was not fully read yet but
            //  we write into smem for reduction
            __syncthreads();
        }
        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2];
        // softmax.template reduce<fmha::Max_>(p_max);
        softmax.reduce_max(p_max);

        // Make sure we are done reading shared memory.
        //__syncthreads();

        // Compute the exponential value.
        softmax.apply_exp(p_max);

        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2];
        softmax.reduce_sum(p_sum);

        // Finalize softmax on the accumulators of P^T.
        softmax.scale(p_sum);
        using Frag_p = fmha::Fragment_a<Traits, fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        if (IS_TRAINING)
        {
            auto encode_dropout = [](bool keep, float val) { return keep ? val : -val; };
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::MMAS_M; mi++)
            {
#pragma unroll
                for (int ii = 0; ii < 2; ii++)
                {
#pragma unroll
                    for (int ni = 0; ni < Mma_tile_p::MMAS_N; ni++)
                    {
                        float4 tmp = uniform4(ph());
                        // We encode the dropout pattern in the sign bit of
                        //  the non-negative softmax to distinguish from pre-existing zeros
                        softmax.elt_[2 * mi + ii][4 * ni + 0]
                            = encode_dropout(tmp.x <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 0]);
                        softmax.elt_[2 * mi + ii][4 * ni + 1]
                            = encode_dropout(tmp.y <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 1]);
                        softmax.elt_[2 * mi + ii][4 * ni + 2]
                            = encode_dropout(tmp.z <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 2]);
                        softmax.elt_[2 * mi + ii][4 * ni + 3]
                            = encode_dropout(tmp.w <= params.p_dropout, softmax.elt_[2 * mi + ii][4 * ni + 3]);
                    }
                }
            }
            softmax.pack(frag_p);
            gmem_s.store(frag_p, mask);
            // gmem_s.store(softmax.elt_, mask);
            gmem_s.move();
        }
        else
        {
            softmax.pack(frag_p);
        }

#pragma unroll
        for (int ki = 0; ki < Mma_tile_o::MMAS_K; ki++)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_o::MMAS_M; mi++)
            {
#pragma unroll
                for (int ii = 0; ii < Frag_p::NUM_REGS; ii++)
                {
                    //"Apply" the dropout.
                    frag_p[ki][mi].reg(ii) = fmha::hmul2(frag_p[ki][mi].reg(ii), params.scale_dropout);
                    frag_p[ki][mi].reg(ii) = fmha::hrelu2(frag_p[ki][mi].reg(ii));
                }
            }
        }

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        // Make sure we have the LDGDEPBAR in place.
        fmha::ldgdepbar<USE_LDGSTS_Q>();

// Do this part of O = P^T * V^T.
#pragma unroll
        for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
        {
            fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
        }

// Loop over MMAS_M.
#pragma unroll
        for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
        {

            // Swizzle the elements and do the final reduction.
            smem_o.store(acc_o, ii);

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

        // Move to the next part of the output.
        gmem_o.move();
        smem_k.load(frag_k[0], 0);

        // Commit the values for Q into shared memory.
        // if( loop + Cta_tile_p::M < Cta_tile_p::N ) {
        if (l < steps - 1)
        {
            gmem_q.commit(smem_q);
            __syncthreads();
            smem_q.load(frag_q[0], 0);
        }
        /*
        // Make sure we are reading from the correct buffer.
        if( USE_LDGSTS_Q ) {
            smem_q.move_to_next_read_buffer();
        }

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS_Q, 1>();
        __syncthreads();

        // Trigger the loads for the values of Q for the next iteration.
        smem_q.load(frag_q[0], 0);
        */

    } // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool IS_TRAINING, typename Params>
inline __device__ void device_1xN(Params const& params, int const num_full_heads, int const num_main_groups,
    int const main_group_size, int const main_steps, int const rest_steps)
{

    constexpr int STEPS = Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M;
    int const tidx_global = blockIdx.x * gridDim.x + threadIdx.x;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph(std::get<0>(seeds), tidx_global, std::get<1>(seeds));
    for (int it = 0; it < num_full_heads; it++)
    {
        int const bidx = it * gridDim.x + blockIdx.x;
        int const bidh = bidx % params.h;
        int const bidb = bidx / params.h;
        fused_multihead_attention::device_1xN_<Kernel_traits, IS_TRAINING>(params, bidb, bidh, 0, STEPS, ph);
    }
    if (main_group_size == 0)
        return;
    int const head_offset = num_full_heads * gridDim.x;

    if (blockIdx.x < main_group_size * num_main_groups)
    {
        // process within heads
        int const group = blockIdx.x % num_main_groups;
        int const bidx = blockIdx.x / num_main_groups;
        int const bidh = (head_offset + bidx) % params.h;
        int const bidb = (head_offset + bidx) / params.h;
        int const offset = group * main_steps;
        fused_multihead_attention::device_1xN_<Kernel_traits, IS_TRAINING>(params, bidb, bidh, offset, main_steps, ph);
    }
    else
    {
        // process across heads
        int const bidx = blockIdx.x - main_group_size * num_main_groups;
        int const offset = num_main_groups * main_steps;
        int const total_heads = params.b * params.h;
        int const rest_ctas = gridDim.x - main_group_size * num_main_groups;
        for (int it = head_offset + bidx; it < total_heads; it += rest_ctas)
        {
            int const bidh = it % params.h;
            int const bidb = it / params.h;
            fused_multihead_attention::device_1xN_<Kernel_traits, IS_TRAINING>(
                params, bidb, bidh, offset, rest_steps, ph);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
