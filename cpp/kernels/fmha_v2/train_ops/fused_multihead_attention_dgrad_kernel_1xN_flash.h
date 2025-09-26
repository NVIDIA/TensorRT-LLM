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
#include "smem_tile_dq.h"
#include <fmha/gemm.h>
#include <fmha/kernel_traits.h>

namespace fused_multihead_attention
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params, typename Prng>
inline __device__ void compute_dq_dk_dv_1xN_kv_inner_loop_(Params const& params, Prng& ph, int kv_offset)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // For that kernel, each CTA is assigned a portion of the K/V dimension. The size assigned
    // to each CTA is given by Kernel_traits::Cta_tile_p::N. Each CTA computes dQ, dK and dV
    // while it iterates over the Q dimension. The accumulation for dQ happens in global memory
    // and uses FP32 atomic instructions (multiple CTAs contribute to dQ). The accumulation of
    // dK and dV is done locally and kept on the SM (in registers or shared memory).
    //
    // It means that each CTA executes a loop in the Q dimension. Before the loop, it fetches
    // blocks of K and V that are kept on the SM during the entire loop. At each iteration of the
    // loop, each CTA loads the corresponding blocks of Q and dO and assemble the different tiles
    // on the SM.
    //
    // The first thing we do is to recompute P = Q * K^T and reassemble the intermediate Softmax
    // tile that is built at each iteration along the K/V dimension of the forward pass in Flash
    // Attention. That partial reconstruction is called P' in the sequel. It leads to a first type
    // of batched GEMM that is parametrized on Cta_tile_p.
    //
    // We also build the output tensor dQ = dP * K. It's a GEMM parametrized on
    // Kernel_traits::Cta_tile_o. However, to be able to compute that tile, we need to assemble
    // dP = P' * (dP' - reduce_sum(O * dO)) where P' is the partial reconstruction of Softmax from
    // the forward pass, dP' = dropout(dS) and dS is dS = dO * V^T (i.e. the output error after
    // Softmax). Note that P' can be reconstructed using the sums and max stored during the forward
    // pass and reduce_sum(O * dO) is pre-computed in a separate kernel.
    //
    // Finally, we assemble dK = dP^T * Q and dV = S^T * dO. We keep those two tiles on the SM
    // during the entire main loop and only output them at the end.
    //

    // The A data type of all GEMMs
    using A_data_type = typename Kernel_traits::Traits_p::A_type;

    // The description of the CTA tile for the GEMM to compute P.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the GEMMs to compute dS and dQ.
    using Cta_tile_ds_dq = typename Kernel_traits::Cta_tile_o;
    // The description of the CTA tile for the GEMMs to compute dK and dV.
    using Cta_tile_dk_dv = typename Traits::template Cta_tile_extd<Cta_tile_p::N, // STEPK
        Cta_tile_p::K,                                                            // D
        Cta_tile_p::M,                                                            // STEPQ
        Cta_tile_p::VALID_K,                                                      // VALID_N
        Cta_tile_p::M,                                                            // VALID_K
        Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;

    // The MMA tile for the GEMM to compute P = Q * K^T.
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the GEMMs to compute dS = dO * V^T and dQ = dP * K.
    using Mma_tile_ds_dq = typename Traits::template Mma_tile<Cta_tile_ds_dq>;
    // The MMA tile for the GEMMs to compute dK = dP^T * Q and dV = S^T * dO.
    using Mma_tile_dk_dv = typename Traits::template Mma_tile<Cta_tile_dk_dv>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to load Q for P = Q * K^T.
    // using Smem_tile_q_a = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_q_a = fmha::Smem_tile_a<Traits, Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The shared memory tile to load Q for dK = dP^T * Q.
    using Smem_tile_q_b = fmha::Smem_tile_b<Traits, Cta_tile_dk_dv, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // Make sure the buffers have the same sizes.
    static_assert(Smem_tile_q_a::BYTES_PER_TILE == Smem_tile_q_b::BYTES_PER_TILE, "");

    // The global memory tile to load K.
    using Gmem_tile_kt = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to load K for P = Q * K^T.
    using Smem_tile_kt_b = typename Kernel_traits::Smem_tile_k;
    // The shared memory tile to load K for dQ = dP * K.
    using Smem_tile_k_b = typename Kernel_traits::Smem_tile_v;

    // Make sure the buffers have the same sizes.
    static_assert(Smem_tile_kt_b::BYTES_PER_TILE == Smem_tile_k_b::BYTES_PER_TILE, "");

    // The global memory tile to load V^T. We use *_tile_k to transpose on-the-fly.
    using Gmem_tile_vt = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to load V^T for S = dO * V^T.
    using Smem_tile_vt_b = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load dO.
    using Gmem_tile_do = Gmem_tile_dout<Traits, Cta_tile_p>;
    // The shared memory tile to load dO for dS = dO * dV^T.
    using Smem_tile_do_a = fmha::Smem_tile_a<Traits, Cta_tile_p, fmha::Row, Gmem_tile_do::BYTES_PER_LDG, 2>;

    // The description of the CTA tile to load dO as the B fragment of dV = S^T * dO.
    using Cta_tile_do_b = typename Traits::template Cta_tile_extd<Cta_tile_p::N,
        Cta_tile_p::K,       // D
        Cta_tile_p::M,       // STEPQ
        Cta_tile_p::VALID_K, // valid_n
        Cta_tile_p::M,       // valid_k
        Cta_tile_p::WARPS_N, 1, 1>;
    // The shared memory tile to reload dO for dV = S^T * dO.
    using Smem_tile_do_b = fmha::Smem_tile_b<Traits, Cta_tile_do_b, fmha::Row, Gmem_tile_do::BYTES_PER_LDG, 2>;

    // Make sure the buffers have the same sizes.
    static_assert(Smem_tile_do_a::BYTES_PER_TILE == Smem_tile_do_b::BYTES_PER_TILE, "");

    // The global memory tile to store dQ.
    using Gmem_tile_dq = Gmem_tile_dq_red<Traits, Cta_tile_ds_dq>;
    // The shared memory tile to do the reduction before writing to global memory.
    using Smem_tile_dq = Smem_tile_dq_red<Traits, Cta_tile_ds_dq>;

    // The global memory tile to store dK.
    using Gmem_tile_dk = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dK.
    using Smem_tile_dk = Smem_tile_mma_epilogue<Traits, Cta_tile_dk_dv>;

    // The global memory tile to store dV.
    using Gmem_tile_dv = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dV.
    using Smem_tile_dv = Smem_tile_mma_epilogue<Traits, Cta_tile_dk_dv>;

    // The shared memory tile to transpose S.
    using Smem_tile_st = Smem_tile_mma_transposed<Traits, Cta_tile_p>;

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

    // Make sure we do not try to use LDGSTS.
    static_assert(USE_LDGSTS == 0, "");

    // Shared memory.
    extern __shared__ char smem_[];

    // The K/V step. Map it to blockIdx.x to increase the hit rate in cache.
    int const bidkv = blockIdx.x + kv_offset;
    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.z;
    // The thread index.
    int const tidx = threadIdx.x;

    // Early exit.
    Block_info_padded<Kernel_traits::THREADS, Kernel_traits::SEQUENCES_INTERLEAVED> const binfo(
        params, bidb, bidh, tidx);
    if (binfo.stop_early())
    {
        return;
    }

    // The mask.
    fmha::Mask<Traits, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);
    // Determine whether causal mask is applied or not.
    bool const causal_mask = Kernel_traits::MASK_VERSION == 3;

    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    int begin = causal_mask ? bidkv * Cta_tile_p::N / Cta_tile_p::M : 0;

    // Allocate the global memory tile loader for Q, K, V and dO.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    Gmem_tile_kt gmem_kt(params, 1, binfo, tidx);
    Gmem_tile_vt gmem_vt(params, 2, binfo, tidx);
    Gmem_tile_do gmem_do(params, binfo, tidx);
    Gmem_tile_dq gmem_dq(params, binfo, tidx);

    // Move to the correct position for K and V.
    gmem_kt.move(bidkv);
    gmem_vt.move(bidkv);
    // Move to the correct position for Q and dO.
    gmem_do.move_to(begin);
    gmem_q.move_to(begin);

    // The array that contains LSE values for softmax.
    fmha::Softmax_statistics<Traits, Cta_tile_p> lse_array(params, params.lse_ptr, binfo, tidx);
    // The array that contains the sums for softmax.
    fmha::Softmax_statistics<Traits, Cta_tile_p> sum_array(params, params.softmax_sum_ptr, binfo, tidx);

    // Offsets to store the different tiles in shared memory.
    int const SMEM_OFFSET_dO = 0;
    int const SMEM_OFFSET_Q = SMEM_OFFSET_dO + Smem_tile_do_a::BYTES_PER_TILE;
    int const SMEM_OFFSET_K = SMEM_OFFSET_Q + Smem_tile_q_a ::BYTES_PER_TILE;
    int const SMEM_OFFSET_V = SMEM_OFFSET_K + Smem_tile_kt_b::BYTES_PER_TILE;
    int const SMEM_OFFSET_dQ = SMEM_OFFSET_V + Smem_tile_vt_b::BYTES_PER_TILE;
    int const SMEM_OFFSET_S = SMEM_OFFSET_dQ + Smem_tile_dq ::BYTES_PER_TILE;
    int const SMEM_OFFSET_dP = SMEM_OFFSET_S + Smem_tile_st ::BYTES_PER_TILE;

    // | dO_a | Q_a | K_b | V | dQ | St | dP |
    // | dO_b | Q_b | Kt_b|

    // Make sure the assumptions hold regarding the size of the tiles.
    static_assert(Smem_tile_do_a::BYTES_PER_TILE == Smem_tile_do_b::BYTES_PER_TILE, "");
    static_assert(Smem_tile_q_a ::BYTES_PER_TILE == Smem_tile_q_b ::BYTES_PER_TILE, "");
    static_assert(Smem_tile_kt_b::BYTES_PER_TILE == Smem_tile_k_b ::BYTES_PER_TILE, "");

    // Make sure we can reduce dQ for S and dP.
    static_assert(Smem_tile_dq::BYTES_PER_TILE >= Smem_tile_st::BYTES_PER_TILE * 2, "");

    // Allocate the shared memory tile loaders.
    Smem_tile_do_a smem_do_a(&smem_[SMEM_OFFSET_dO], tidx);
    Smem_tile_do_b smem_do_b(&smem_[SMEM_OFFSET_dO], tidx);
    Smem_tile_q_a smem_q_a(&smem_[SMEM_OFFSET_Q], tidx);
    Smem_tile_q_b smem_q_b(&smem_[SMEM_OFFSET_Q], tidx);
    Smem_tile_kt_b smem_kt_b(&smem_[SMEM_OFFSET_K], tidx);
    Smem_tile_k_b smem_k_b(&smem_[SMEM_OFFSET_K], tidx);
    Smem_tile_vt_b smem_vt_b(&smem_[SMEM_OFFSET_V], tidx);
    Smem_tile_dq smem_dq(&smem_[SMEM_OFFSET_dQ], tidx);
    Smem_tile_st smem_st(&smem_[SMEM_OFFSET_S], tidx);
    Smem_tile_st smem_dp(&smem_[SMEM_OFFSET_dP], tidx);

    // Trigger the first loads for Q, K^T, dO and V^T.
    gmem_q.load(smem_q_a);
    gmem_kt.load(smem_kt_b);
    gmem_do.load(smem_do_a);
    gmem_vt.load(smem_vt_b);

    // Commit the data to shared memory.
    gmem_q.commit(smem_q_a);
    gmem_kt.commit(smem_kt_b);
    gmem_do.commit(smem_do_a);
    gmem_vt.commit(smem_vt_b);

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, 0, bidb, tidx);

    // The type of accumulators.
    using Accumulator_type = typename Traits::Accumulator_type;

    // Declare the accumulators for gemm dV = S^T * dO.
    fmha::Fragment_accumulator<Traits> acc_dv[Mma_tile_dk_dv::MMAS_M][Mma_tile_dk_dv::VALID_MMAS_N];
    fmha::Clear_accumulator<Accumulator_type, Cta_tile_dk_dv::WARPS_K>::apply(acc_dv);

    // Declare the accumulators for gemm: dK = dP^T * Q
    fmha::Fragment_accumulator<Traits> acc_dk[Mma_tile_dk_dv::MMAS_M][Mma_tile_dk_dv::VALID_MMAS_N];
    fmha::Clear_accumulator<Accumulator_type, Cta_tile_dk_dv::WARPS_K>::apply(acc_dk);

    // Make sure the data is in shared memory.
    __syncthreads();

    typename Smem_tile_q_a::Fragment frag_q[2][Mma_tile_p::MMAS_M];
    typename Smem_tile_kt_b::Fragment frag_kt[2][Mma_tile_p::MMAS_N];

    // Load the first fragments for Q.
    smem_q_a.load(frag_q[0], 0);

    // Load the first fragments for K^T.
    smem_kt_b.load(frag_kt[0], 0);

    // Iterate over the sequence.
    int const STEPS = (binfo.actual_seqlen + Cta_tile_p::M - 1) / Cta_tile_p::M;
#pragma unroll 1
    for (int step = begin; step < STEPS; ++step)
    {

        //
        // Trigger the load for the next Q/dO values.
        //

        if (step + 1 < STEPS)
        {
            gmem_q.move_to(step + 1);
            gmem_do.move_to(step + 1);
            gmem_q.load();
            gmem_do.load();
        }

        //
        // Compute P = Q * K^T.
        //

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

// Perform the first GEMM: P = Q * K^T.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the Q values.
            smem_q_a.load(frag_q[ki & 1], ki);
            smem_kt_b.load(frag_kt[ki & 1], ki);

            // Do the math for the values already in registers.
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_kt[(ki - 1) & 1]);
            }
        }

        // Load the first fragments for dO from shared memory in advance.
        typename Smem_tile_do_a::Fragment frag_do[2][Mma_tile_p::MMAS_M];
        smem_do_a.load(frag_do[0], 0);

        // Prepare data for dS = V * dO. Load the first fragments for V from shared memory.
        typename Smem_tile_vt_b::Fragment frag_vt[2][Mma_tile_p::MMAS_N];
        smem_vt_b.load(frag_vt[0], 0);

        // Load the fragments for K.
        typename Smem_tile_k_b::Fragment frag_k[2][Mma_tile_ds_dq::VALID_MMAS_N];
        smem_k_b.load(frag_k[0], 0);

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_kt[(ki - 1) & 1]);
            }
        }

        //
        // Reconstruct S = Softmax(P).
        //

        // Load softmax lse. TODO: Pre-scale LSE with LOG2E.
        float lse_regs[2 * Mma_tile_p::MMAS_M];
        lse_array.load(step);
        for (int ii = 0; ii < 2 * Mma_tile_p::MMAS_M; ii++)
        {
            lse_regs[ii] = lse_array.lm_[ii];
        }

        // Load softmax sum.
        float softmax_sum[2 * Mma_tile_p::MMAS_M];
        sum_array.load(step);
        for (int ii = 0; ii < 2 * Mma_tile_p::MMAS_M; ii++)
        {
            softmax_sum[ii] = sum_array.lm_[ii];
        }

        // Load the mask.
        // mask.load(step);

        // Apply softmax to P to produce S. It's needed to compute dV = S^T * dO.

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);

        // Move mask to the correct position.
        mask.load(step, bidkv);

        // Apply the mask.
        if (params.has_alibi)
        {
            softmax.apply_mask_alibi(mask, bidh, params.alibi_params, params.fscale_bmm1);
            softmax.apply_exp(lse_regs);
        }
        else
        {
            softmax.apply_mask(mask);
            softmax.apply_scale_exp(lse_regs, params.fscale_bmm1);
        }

        if (step + 1 < STEPS)
        {
            // Move the shared memory pointers to the correct locations to commit data.
            smem_q_a.move_to_next_write_buffer();
            smem_q_b.move_to_next_write_buffer();
            smem_do_a.move_to_next_write_buffer();
            smem_do_b.move_to_next_write_buffer();

            // Commit data (Q and dO) to shared memory (except for the last step).
            gmem_q.commit(smem_q_a);
        }

        if (params.has_dropout)
        {
            // for warp 1xN
            static_assert(Cta_tile_p::WARPS_M == 1);
            // 32 means the number of thread per warp.
            unsigned int warp_idx = threadIdx.x / 32;
            // 16 means each warp will handle 16 x 16 elements in current MMA computation.
            unsigned int block_col_idx = bidkv * Cta_tile_p::N / 16 + warp_idx;
            unsigned long long philox_subsequence = step * (binfo.actual_seqlen / 16) + block_col_idx;

            auto encode_dropout = [](bool keep, float val) { return keep ? val : -val; };

#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::MMAS_M; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < Mma_tile_p::MMAS_N; ni++)
                {

                    uint16_t* rand_arr = (uint16_t*) &ph(philox_subsequence + ni * Cta_tile_p::WARPS_N);

#pragma unroll
                    for (int jj = 0; jj < 2; ++jj)
                    {
#pragma unroll
                        for (int ii = 0; ii < 4; ii++)
                        {
                            // We encode the dropout pattern in the sign bit of the non-negative
                            // softmax to distinguish from pre-existing zeros
                            softmax.elt_[mi * 2 + jj][4 * ni + ii] = rand_arr[4 * jj + ii] <= params.p_dropout_16bit
                                ? softmax.elt_[mi * 2 + jj][4 * ni + ii]
                                : -softmax.elt_[mi * 2 + jj][4 * ni + ii];
                        } // jj
                    }     // ii
                }         // ni
            }             // mi
        }                 // has_dropout

        // Pack the output of softmax to FP16x2 to limit the shared memory footprint.
        uint4 regs_s[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        softmax.pack(regs_s);
        smem_st.store(regs_s);

        //
        // Compute dP = dO * V^T.
        //

        // Initialize the accumulators to -sum.
        fmha::Fragment_accumulator<Traits> acc_dp[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::MMAS_N; ++ni)
            {
#pragma unroll
                for (int ii = 0; ii < 8; ++ii)
                {
                    acc_dp[mi][ni].elt(ii) = -softmax_sum[mi * 2 + ((ii / 2) % 2)];
                }
            }
        }

// Do this part of dP = dO * V^T.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the next series of dO/V^T values.
            smem_do_a.load(frag_do[ki & 1], ki);
            smem_vt_b.load(frag_vt[ki & 1], ki);

            // Do the math for the values already in registers.
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_dp, frag_do[(ki - 1) & 1], frag_vt[(ki - 1) & 1]);
            }
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_dp, frag_do[(ki - 1) & 1], frag_vt[(ki - 1) & 1]);
            }
        }

        float const scale_softmax_rp_droput = reinterpret_cast<float const&>(params.scale_softmax) * params.rp_dropout;
        // double check the accuracy and performance.
        auto pointwise_mult = [](float p, float dp, float softmax_sum, float scale, bool drop)
        { return p * ((!drop) || p >= 0.f ? dp : softmax_sum) * scale; };
// Populate the softmax object with correct values.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::MMAS_N; ++ni)
            {
                // softmax is S = dropout (P'), have both positive and negative values
                softmax.elt_[2 * mi + 0][4 * ni + 0] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 0],
                    acc_dp[mi][ni].elt(0), softmax_sum[2 * mi + 0], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 0][4 * ni + 1] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 1],
                    acc_dp[mi][ni].elt(1), softmax_sum[2 * mi + 0], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 0][4 * ni + 2] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 2],
                    acc_dp[mi][ni].elt(4), softmax_sum[2 * mi + 0], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 0][4 * ni + 3] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 3],
                    acc_dp[mi][ni].elt(5), softmax_sum[2 * mi + 0], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 1][4 * ni + 0] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 0],
                    acc_dp[mi][ni].elt(2), softmax_sum[2 * mi + 1], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 1][4 * ni + 1] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 1],
                    acc_dp[mi][ni].elt(3), softmax_sum[2 * mi + 1], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 1][4 * ni + 2] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 2],
                    acc_dp[mi][ni].elt(6), softmax_sum[2 * mi + 1], scale_softmax_rp_droput, params.has_dropout);
                softmax.elt_[2 * mi + 1][4 * ni + 3] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 3],
                    acc_dp[mi][ni].elt(7), softmax_sum[2 * mi + 1], scale_softmax_rp_droput, params.has_dropout);
            }
        }

        // Pack dP to FP16x2 for the next GEMM and to store to shared memory (for dK).
        uint4 regs_dp[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        softmax.pack(regs_dp);
        smem_dp.store(regs_dp);

        // Make sure the dimensions match.
        static_assert(Mma_tile_ds_dq::MMAS_M == Mma_tile_p::MMAS_M, "");
        static_assert(Mma_tile_ds_dq::MMAS_K == Mma_tile_p::MMAS_N, "");

        // Declare the accumulators for the 2nd gemm: Compute dQ = dP * K.
        fmha::Fragment_accumulator<Traits> acc_dq[Mma_tile_ds_dq::MMAS_M][Mma_tile_ds_dq::VALID_MMAS_N];
        fmha::Clear_accumulator<Accumulator_type, Cta_tile_ds_dq::WARPS_K>::apply(acc_dq);

// Do this part of dQ = dP * K.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_ds_dq::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the next series of Q values.
            smem_k_b.load(frag_k[ki & 1], ki);

            // Remap the fragments.
            fmha::Fragment_a<Traits, fmha::Row> frag_dp[Mma_tile_ds_dq::MMAS_M];
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi)
            {
                frag_dp[mi].reg(0) = regs_dp[mi][ki - 1].x;
                frag_dp[mi].reg(1) = regs_dp[mi][ki - 1].z;
                frag_dp[mi].reg(2) = regs_dp[mi][ki - 1].y;
                frag_dp[mi].reg(3) = regs_dp[mi][ki - 1].w;
            }

            // Do the math for the values already in registers.
            fmha::gemm(acc_dq, frag_dp, frag_k[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            fmha::Fragment_a<Traits, fmha::Row> frag_dp[Mma_tile_ds_dq::MMAS_M];
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi)
            {
                frag_dp[mi].reg(0) = regs_dp[mi][Mma_tile_ds_dq::MMAS_K - 1].x;
                frag_dp[mi].reg(1) = regs_dp[mi][Mma_tile_ds_dq::MMAS_K - 1].z;
                frag_dp[mi].reg(2) = regs_dp[mi][Mma_tile_ds_dq::MMAS_K - 1].y;
                frag_dp[mi].reg(3) = regs_dp[mi][Mma_tile_ds_dq::MMAS_K - 1].w;
            }
            fmha::gemm(acc_dq, frag_dp, frag_k[(Mma_tile_ds_dq::MMAS_K - 1) & 1]);
        }

        if (step + 1 < STEPS)
        {
            // Commit data (Q and dO) to shared memory (except for the last step).
            gmem_do.commit(smem_do_a);
        }

// Perform the reduction of dQ. Use shared memory and global memory atomics.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi)
        {
            smem_dq.store(acc_dq[mi], mi);
        }

        //
        // Do the computations for dV = S^T * dO.
        //

        // Load all the fragments for S.
        typename Smem_tile_st::Fragment frag_st[Mma_tile_dk_dv::MMAS_K][Mma_tile_dk_dv::MMAS_M];
        // Make sure smem_st's store has completed.
        __syncthreads();
        smem_st.load(frag_st);

        // Load the first fragment for dO.
        typename Smem_tile_do_b::Fragment frag_do_b[2][Mma_tile_dk_dv::VALID_MMAS_N];
        smem_do_b.load(frag_do_b[0], 0);
        static_assert(Mma_tile_dk_dv::MMAS_K == 1);

        if (params.has_dropout)
        {
#pragma unroll
            for (int ki = 0; ki < Mma_tile_dk_dv::MMAS_K; ki++)
            {
#pragma unroll
                for (int mi = 0; mi < Mma_tile_dk_dv::MMAS_M; mi++)
                {
#pragma unroll
                    for (int ii = 0; ii < Smem_tile_st::Fragment::NUM_REGS; ii++)
                    {
                        frag_st[ki][mi].reg(ii) = fmha::relu2<A_data_type>(frag_st[ki][mi].reg(ii));
                    }
                }
            }
        }

// Do the first part of the GEMM.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_dk_dv::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the next series of MMAs.
            smem_do_b.load(frag_do_b[ki & 1], ki);

            // Do the math for the values already in registers.
            fmha::gemm(acc_dv, frag_st[(ki - 1)], frag_do_b[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dk_dv::MMAS_K;
            fmha::gemm(acc_dv, frag_st[(ki - 1)], frag_do_b[(ki - 1) & 1]);
        }

        if (step + 1 < STEPS)
        {

            // Move the shared memory buffers for Q/dO.
            smem_do_a.move_to_next_read_buffer();
            smem_do_b.move_to_next_read_buffer();
        }

        //
        // Do the computations for dK = dP^T * Q.
        //

        // Load the entire dP^T from shared memory.
        typename Smem_tile_st::Fragment frag_dpt[Mma_tile_dk_dv::MMAS_K][Mma_tile_dk_dv::MMAS_M];
        smem_dp.load(frag_dpt);

        // Reload the first fragments of Q from shared memory.
        typename Smem_tile_q_b::Fragment frag_q_b[2][Mma_tile_dk_dv::VALID_MMAS_N];
        smem_q_b.load(frag_q_b[0], 0);

// Do the math. Compute dK = dP^T * Q.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_dk_dv::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the next series of Q values.
            smem_q_b.load(frag_q_b[ki & 1], ki);

            // Do the math for the values already in registers.
            fmha::gemm(acc_dk, frag_dpt[(ki - 1)], frag_q_b[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dk_dv::MMAS_K;
            fmha::gemm(acc_dk, frag_dpt[(ki - 1)], frag_q_b[(ki - 1) & 1]);
        }

        // Make sure we can write to shared memory and that dQ is in shared memory.
        __syncthreads();

        if (step + 1 < STEPS)
        {

            // Move the shared memory buffers for Q/dO.
            smem_q_a.move_to_next_read_buffer();
            smem_q_b.move_to_next_read_buffer();

            // Load the first fragments for Q.
            smem_q_a.load(frag_q[0], 0);
            // Load the first fragments for K^T.
            smem_kt_b.load(frag_kt[0], 0);
        }

// Load dQ from shared memory to update the running sums (using atomics).
// const float scale_bmm1_rp_dropout = params.fscale_bmm1 * params.rp_dropout;
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi)
        {
            // #pragma unroll
            for (int ii = 0; ii < Gmem_tile_dq::REDS; ++ii)
            {
                float dq_out = 0.f;
                smem_dq.load(dq_out, ii);
                // dq_out *= scale_bmm1_rp_dropout;
                gmem_dq.store(step, mi, ii, dq_out);
            }
        }

    } // END of the loop.

    if (params.has_dropout)
    {
        for (int mi = 0; mi < Mma_tile_dk_dv::MMAS_M; mi++)
        {
            for (int ni = 0; ni < Mma_tile_dk_dv::VALID_MMAS_N; ni++)
            {
                acc_dv[mi][ni].mul(params.rp_dropout);
            }
        }
    }

    // Make sure we can write to shared memory.
    __syncthreads();

    // The parameters to store the output.
    Qkv_params dqkv_params;
    dqkv_params.qkv_ptr = params.dqkv_ptr;
    dqkv_params.qkv_stride_in_bytes = params.qkv_stride_in_bytes;
    dqkv_params.h = params.h;

    // The global memory tiles to store dK and dV.
    Gmem_tile_dk gmem_dk(dqkv_params, 1, binfo, tidx);
    Gmem_tile_dv gmem_dv(dqkv_params, 2, binfo, tidx);

    // Position to correct K/V step.
    gmem_dk.move(bidkv);
    gmem_dv.move(bidkv);

    // Store dK to shared memory to swizzle the output.
    Smem_tile_dk smem_dk(&smem_[Smem_tile_dv::BYTES_PER_TILE], tidx);
    smem_dk.store(acc_dk);

    // Store dV to shared memory to swizzle the output.
    Smem_tile_dv smem_dv(&smem_[0], tidx);
    smem_dv.store(acc_dv);

    // Make sure the data is in shared memory.
    __syncthreads();

    // Load from shared memory and output to global memory.
    uint4 dk_out[Smem_tile_dk::NUM_LDS];
    smem_dk.load(dk_out);
    gmem_dk.store(dk_out);

    uint4 dv_out[Smem_tile_dv::NUM_LDS];
    smem_dv.load(dv_out);
    gmem_dv.store(dv_out);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void compute_dq_dk_dv_1xN_kv_inner_loop(Params const& params, int kv_iterations = 1)
{
    // Make sure that the fwd and bwd to generate the same dropout pattern.
    // The block index for the heads.
    int const bidh = blockIdx.z;
    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The thread index.
    int const tidx = threadIdx.x;

    // auto seeds = at::cuda::philox::unpack(params.philox_args);
    uint64_t* s_ptr = reinterpret_cast<uint64_t*>(params.seed_ptr);
    uint64_t seed = s_ptr[0];
    uint64_t offset = s_ptr[1];

    // if (threadIdx.x == 0) printf("back seed = %d, %d\n", (int)seed, (int)offset);

    // TODO: offset is not the same with fprop, the WAR is using the seed/offset passed from fprop.
    Philox ph(seed, 0, offset + (bidb * params.h + bidh) * 32 + tidx % 32);

    for (int it = 0; it < kv_iterations; it++)
    {
        int kv_offset = it * gridDim.x;
        fused_multihead_attention::compute_dq_dk_dv_1xN_kv_inner_loop_<Kernel_traits, Params>(params, ph, kv_offset);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The index will be used in shfl function to change the order of the elements.
// original order: [0, 4, 1, 5,  2,  6,  3, 7]
// expected order: [0, 1, 2, 3,  4,  5,  6, 7]
//      offset is: [0, 1, 2, 3, -3, -2, -1, 0]
static __device__ int index_cal(int quad_lane_id)
{
    return (quad_lane_id - (quad_lane_id >= 4) * 7);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void convert_dq_to_16bits(Params const& params)
{

    // The instruction traits for dQ.
    using Traits_dq = typename Kernel_traits::Traits_o;
    // The description of the CTA tile for the GEMMs to compute dQ.
    using Cta_tile_dq = typename Kernel_traits::Cta_tile_o;

    // The A data type
    using A_data_type = typename Kernel_traits::Traits_o::A_type;

    // Each thread fetches 4x float and produces 4x 16-bit numbers in 2 regs.

    // The number of features per thread.
    enum
    {
        FEATURES_PER_THREAD = 2
    };

    // The number of threads per head.
    enum
    {
        THREADS_PER_TOKEN = Cta_tile_dq::N / FEATURES_PER_THREAD
    };

    // The valid number of threads per head.
    enum
    {
        VALID_THREADS_PER_TOKEN = Cta_tile_dq::VALID_N / FEATURES_PER_THREAD
    };

    // Next head's offset: head_interleaved [b, s, h, 3, d]
    static constexpr int NEXT_HEAD_OFFSET_FACTOR = Kernel_traits::HEADS_INTERLEAVED ? 3 : 1;

    // DEBUG.
    static_assert(
        (Cta_tile_dq::N == 64 || Cta_tile_dq::N == 128) && (THREADS_PER_TOKEN == 32 || THREADS_PER_TOKEN == 64), "");
    // END OF DEBUG.

    // The token index. Each CTA works on a different token.
    int const bidt = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    // The 1st head written by that thread.
    int head = tidx / THREADS_PER_TOKEN;
    // The feature.
    int feat = tidx % THREADS_PER_TOKEN;

    // Compute the input offset. Layout is HEADs x TOTAL_S x DIM_PER_HEAD. 4x FP32 per LDG.
    int64_t input_offset = head * params.total_s * Cta_tile_dq::N + bidt * Cta_tile_dq::N;
    // The input pointer.
    float const* input_ptr = reinterpret_cast<float const*>(params.dq_acc_ptr);
    // Take the feature/channel into account.
    input_ptr += input_offset + feat * FEATURES_PER_THREAD;

    // Compute the output offset. Layout is TOTAL_S x (3 x HEADs) x DIM_PER_HEAD. 4x 16b per STG.
    int64_t output_offset = bidt * 3 * params.h * Cta_tile_dq::VALID_N;
    // The output pointer.
    uint16_t* output_ptr = reinterpret_cast<uint16_t*>(params.dqkv_ptr);
    // Take the feature/channel into account.
    output_ptr += output_offset;

    // The number of heads written per STG.
    int heads_per_stg = blockDim.x / THREADS_PER_TOKEN;
    // The total number of features per token.
    int features_per_token = params.h * Cta_tile_dq::N;
    // The number of features written per stg.
    int features_per_stg = blockDim.x * FEATURES_PER_THREAD;
    // The number of threads for each head.
    int threads_per_head = blockDim.x / heads_per_stg;
    // Lane id for each head.
    int lane_per_head = tidx % threads_per_head;
    // The number of features written per stg.
    int valid_features_per_stg = heads_per_stg * Cta_tile_dq::VALID_N;

    // thread pos
    int const lane = threadIdx.x % 32;
    int const quad_lane_id = threadIdx.x % 8;
    int const lane_offset = index_cal(quad_lane_id);
    int const src_lane = lane + lane_offset;

    // Iterate over the heads.
    for (int ii = tidx * FEATURES_PER_THREAD; ii < features_per_token; ii += features_per_stg)
    {
        // Load the 2 input floats.
        float2 f2 = *reinterpret_cast<float2 const*>(input_ptr);

        // Convert to 16-bits. TODO: Convert to FP16 or BF16 based on the traits class.
        uint32_t u32 = fmha::float2_to_16bit_2<A_data_type>(f2.x, f2.y);

        u32 = __shfl_sync(0xffffffff, u32, src_lane);

        // Store.
        if (lane_per_head < VALID_THREADS_PER_TOKEN)
        {
            int head_id_per_stg = tidx / threads_per_head;
            uint16_t* local_ptr = output_ptr + head_id_per_stg * Cta_tile_dq::VALID_N * NEXT_HEAD_OFFSET_FACTOR
                + lane_per_head * FEATURES_PER_THREAD;
            reinterpret_cast<uint32_t*>(local_ptr)[0] = u32;
        }

        // Move the input pointer.
        input_ptr += heads_per_stg * params.total_s * Cta_tile_dq::N;
        // Move the output pointer.
        output_ptr += valid_features_per_stg * NEXT_HEAD_OFFSET_FACTOR;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
