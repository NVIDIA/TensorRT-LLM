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
 * St = 16*256*2 = 8KB
 *
 * Shared Memory Layout:
 * |0 |2                           |18K           |26K  |30K   |34K
 * ------------------------------------------------------------------
 * |Q |                          K                             |
 * |Q |                          V                             |
 * |Q |            O               |       S      |  D  | Softmax
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int ROWS, int THREADS_PER_ROW, typename Data_type_, int M, typename Gmem_softmax_sum>
inline __device__ void dot_do_o(
    uint4 const (&do_)[M], uint4 const (&o)[M], float const scale, Gmem_softmax_sum gmem_softmax_d, int tidx)
{
    float sum[M];
    fmha::SumOp<float> sum_op;
#pragma unroll
    for (int mi = 0; mi < M; ++mi)
    {
        sum[mi]
            = fmha::Allreduce<THREADS_PER_ROW>::run(fmha::fma8_in_float<Data_type_>(do_[mi], o[mi]), sum_op) * scale;
    }
    int const dp_sum_row = tidx / THREADS_PER_ROW;
    if ((dp_sum_row < ROWS) && (tidx % THREADS_PER_ROW == 0))
    {
        gmem_softmax_d.store_row(reinterpret_cast<uint32_t const(&)[M]>(sum), dp_sum_row);
    }
}

template <typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o(Params const& params)
{
    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The MMA A data type.
    using A_data_type_ = typename Traits::A_type;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    using Gmem_tile_do = Gmem_tile_dout<Traits, Cta_tile_p>;
    using Gmem_tile_output = Gmem_tile_out<Traits, Cta_tile_p>;
    using Gmem_softmax_sum = Gmem_softmax_sum<Traits, Cta_tile_p>;

    enum
    {
        ROWS = Gmem_tile_do::ROWS
    };

    enum
    {
        THREADS_PER_ROW = Gmem_tile_do::THREADS_PER_ROW
    };

    enum
    {
        LDGS = Gmem_tile_do::LDGS
    };

    int const STEPS = params.s / Cta_tile_p::M;

    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    Block_info_padded<Kernel_traits::THREADS, Kernel_traits::SEQUENCES_INTERLEAVED> const binfo(
        params, bidb, bidh, tidx);
    if (binfo.stop_early())
        return;

    Gmem_tile_do gmem_do(params, binfo, tidx);      // treating d_out as Q
    Gmem_tile_output gmem_out(params, binfo, tidx); // treating d_out as Q
    Gmem_softmax_sum gmem_softmax_d(params, tidx);

    for (int l = blockIdx.z; l < STEPS; l += gridDim.z)
    {
        int const loop = l * Cta_tile_p::M;
        if (loop >= binfo.actual_seqlen)
            break;
        gmem_do.move_to(l);
        gmem_out.move_to(l);
        gmem_softmax_d.move_to(l);

        gmem_do.load();
        gmem_out.load();

        dot_do_o<ROWS, THREADS_PER_ROW, A_data_type_, LDGS, Gmem_softmax_sum>(
            gmem_do.fetch_, gmem_out.fetch_, params.p_dropout, gmem_softmax_d, tidx);
    }
}

template <typename Kernel_traits, typename Params>
inline __device__ void compute_dv_1xN(Params const& params, int const isteps)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dv = typename Traits::template Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M,
        Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;
    // Start of DEBUG
    static_assert(Cta_tile_dv::M == 512 || Cta_tile_dv::M == 384 || Cta_tile_dv::M == 256 || Cta_tile_dv::M == 128
        || Cta_tile_dv::M == 64);
    static_assert(Cta_tile_dv::N == 64 || Cta_tile_dv::N == 128 || Cta_tile_dv::N == 256);
    static_assert(Cta_tile_dv::K == 16);
    // End of DEBUG

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

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dv = typename Traits::template Mma_tile<Cta_tile_dv>;

    // The shared memory tile to swizzle Q.
    // using Smem_tile_do = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_do = fmha::Smem_tile_a<Traits, Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;
    // The shared memory tile to reload Q as fragment b.
    using Smem_tile_dot = fmha::Smem_tile_b<Traits, Cta_tile_dv, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to store dV.
    using Gmem_tile_dv = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dV.
    using Smem_tile_dv = Smem_tile_mma_epilogue<Traits, Cta_tile_dv>;
    static_assert(Smem_tile_dv::NUM_LDS == Gmem_tile_dv::LDGS);
    static_assert(Smem_tile_dv::THREADS_PER_ROW == Gmem_tile_dv::THREADS_PER_ROW);

    // The shared memory tile to transpose S.
    using Smem_tile_st = Smem_tile_mma_transposed<Traits,
        Cta_tile_p>; // should be tile P, since it refers to the input tile

    // using Smem_tile_s  = Smem_tile_mma<Traits, Cta_tile_p>;  // P

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

    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    Block_info_padded<Kernel_traits::THREADS> const binfo(params, bidb, bidh, tidx);
    if (binfo.stop_early())
        return;
    fmha::Mask<Traits, Cta_tile_p, Kernel_traits::VERSION> mask(params, binfo, tidx);

    using Gmem_tile_do = Gmem_tile_dout<Traits, Cta_tile_p>;

    // Allocate the global memory tile loader for Q.
    Gmem_tile_do gmem_do(params, binfo, tidx);  // treating d_out as Q
    Gmem_tile_q gmem_q(params, 0, binfo, tidx); // real Q
    Gmem_tile_k gmem_k(params, 1, binfo, tidx); // real K
    // Allocate the shared memory tile loader for Q.
    Smem_tile_do smem_do(&smem_[0], tidx);
    Smem_tile_dot smem_dot(&smem_[0], tidx);
    Smem_tile_st smem_st(&smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], tidx);

    Smem_tile_q smem_q(
        &smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE], tidx);

    Smem_tile_k smem_kt(&smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE
                            + Smem_tile_q::BYTES_PER_TILE],
        tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_vt(params, 2, binfo, tidx); // treating V as K
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_vt(&smem_[Smem_tile_do::BYTES_PER_TILE], tidx);

    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits::A_type) * 8
    };

    using Gmem_tile_s = Gmem_tile_mma_s<Traits, Cta_tile_p>;

    Gmem_tile_s gmem_s(params, binfo, tidx);

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits, Cta_tile_p, Kernel_traits>;
    // TODO: actually they don't use smem
    Softmax softmax(params,
        &smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE + 2 * Smem_tile_k::BYTES_PER_TILE
            + Smem_tile_q::BYTES_PER_TILE],
        bidb,
        tidx); // TODO

    Softmax softmax_qk(params,
        &smem_[Smem_tile_do::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE + 2 * Smem_tile_k::BYTES_PER_TILE
            + Smem_tile_q::BYTES_PER_TILE],
        bidb,
        tidx); // TODO, will not use Smem for this softmax

    enum
    {
        THREADS_PER_ROW = 32
    };

    enum
    {
        M = Mma_tile_p::MMAS_M
    };

    enum
    {
        N = Mma_tile_p::MMAS_N
    };

    int const STEPS = (binfo.actual_seqlen + Cta_tile_p::M - 1) / Cta_tile_p::M;

    Qkv_params dv_params;
    dv_params.qkv_ptr = params.dqkv_ptr;
    dv_params.qkv_stride_in_bytes = params.qkv_stride_in_bytes;
    dv_params.h = params.h;
    Gmem_tile_dv gmem_dv(dv_params, 2, binfo, tidx);

    // Load over the entire sequence length.
    for (int loop_id = 0; loop_id < isteps; loop_id++)
    { // newloop : new loop
        int newloop = blockIdx.z + gridDim.z * loop_id;
        int const new_loop = newloop * Cta_tile_p::N;
        if (new_loop >= binfo.actual_seqlen)
            // break;
            return;

        gmem_vt.move_to(newloop);
        gmem_k.move_to(newloop);

        // Trigger the loads for QQ.
        gmem_q.load(smem_q);
        // Trigger the loads for KK.
        gmem_k.load(smem_kt);
        // Trigger the loads for Q.
        gmem_do.load(smem_do); // dO
        // Trigger the loads for K.
        gmem_vt.load(smem_vt); // V

        // Push the LDGDEPBAR instruction after the loads for Q, K and V.
        fmha::ldgdepbar<USE_LDGSTS>();

        // Commit the data for Q and K to shared memory.
        gmem_do.commit(smem_do); // dO
        gmem_vt.commit(smem_vt); // V
        gmem_q.commit(smem_q);   // Q
        gmem_k.commit(smem_kt);  // K

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS, 1>();
        __syncthreads();

        // Load the fragments for Q.
        typename Smem_tile_q::Fragment frag_q[2][Mma_tile_p::MMAS_M];
        smem_q.load(frag_q[0], 0);
        // Load the fragments for KK. We keep the data in registers during the entire kernel
        typename Smem_tile_k::Fragment frag_kt[2][Mma_tile_p::MMAS_N];
        smem_kt.load(frag_kt[0], 0);

        // Load the fragments for do.
        typename Smem_tile_do::Fragment frag_do[2][Mma_tile_p::MMAS_M];
        smem_do.load(frag_do[0], 0);

        //      typename Smem_tile_do::Fragment frag_out[2][Mma_tile_p::MMAS_M];

        typename Smem_tile_dot::Fragment frag_dot[2][Mma_tile_dv::MMAS_N];
        static_assert(Smem_tile_dot::Fragment::NUM_REGS == 4);
        static_assert(Mma_tile_dv::MMAS_K == 1);
        smem_dot.load(frag_dot[0], 0);

        // Load the fragments for K. We keep the data in registers during the entire kernel.
        typename Smem_tile_k::Fragment frag_vt[2][Mma_tile_p::MMAS_N];
        smem_vt.load(frag_vt[0], 0);

        // Declare the accumulators for the 2nd gemm. Will be stored every newloop fir each dv tile.
        fmha::Fragment_accumulator<Traits> acc_dv[Mma_tile_dv::MMAS_M][Mma_tile_dv::MMAS_N];
        fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_dv::WARPS_K>::apply(acc_dv);

        for (int l = 0; l < STEPS; l++)
        { //  loop over sequence of sO
            int const loop = l * Cta_tile_p::M;
            if (loop >= binfo.actual_seqlen)
                break;

            // 0. compute Q * k
            fmha::Fragment_accumulator<Traits> acc_qk[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_qk);
#pragma unroll
            for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                // Trigger the load from shared memory for the next series of Q values
                smem_q.load(frag_q[ki & 1], ki);
                smem_kt.load(frag_kt[ki & 1], ki);
                // Do the math for the values already in resitters
                fmha::gemm(acc_qk, frag_q[(ki - 1) & 1], frag_kt[(ki - 1) & 1]);
            }

            // to the final stage of math
            {
                int ki = Mma_tile_p::MMAS_K;
                fmha::gemm(acc_qk, frag_q[(ki - 1) & 1], frag_kt[(ki - 1) & 1]);
            }

            mask.load(l);
            softmax_qk.unpack_noscale(acc_qk); // P

            if (params.has_alibi)
            {
                softmax_qk.apply_mask_alibi(mask, bidh, params.alibi_params);
            }
            else
            {
                softmax_qk.apply_mask(mask);
            }

            // 1. Load S
            uint4 s_regs[M][N]; //: 9.9855e-10to hold P'
            // gmem_s.load(s_regs, mask); // gmem_s is not needed if we don;t output dP'?

            fmha::Fragment_accumulator<Traits> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
            for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                // Trigger the load from shared memory for the next series of Q values.
                smem_do.load(frag_do[ki & 1], ki);
                smem_vt.load(frag_vt[ki & 1], ki);
                // Do the math for the values already in registers.
                fmha::gemm(acc_p, frag_do[(ki - 1) & 1], frag_vt[(ki - 1) & 1]); // dS = dO * V^T
            }

            float lse_regs[2 * M];
            float softmax_sum[2 * M];
            typename fmha::Softmax_statistics<Traits, Cta_tile_p> lse_array(params, params.lse_ptr, binfo, tidx);
            typename fmha::Softmax_statistics<Traits, Cta_tile_p> sum_array(
                params, params.softmax_sum_ptr, binfo, tidx);
            lse_array.load(l);
            sum_array.load(l);
            for (int ii = 0; ii < 2 * M; ii++)
            {
                lse_regs[ii] = lse_array.lm_[ii];
                softmax_sum[ii] = sum_array.lm_[ii];
            }

            // float2 scale_bmm1;
            softmax_qk.apply_scale_exp(lse_regs, params.fscale_bmm1);
            // apply 1/sum, the result is P'
            // softmax_qk.scale(sum_regs);     // TODO: check whether it is the correct function

            softmax_qk.template pack<__half>(s_regs);

            // 2. Store s * dmask to smem for transpose
            smem_st.store(s_regs);

            // Do the final stage of math.
            {
                int ki = Mma_tile_p::MMAS_K;
                fmha::gemm(acc_p, frag_do[(ki - 1) & 1], frag_vt[(ki - 1) & 1]);
            }
            // Trigger the load for the next Q values.
            //  We're using double buffering, so reading qt is safe
            // if( loop + Cta_tile_p::M < Cta_tile_p::N )
            if (l < STEPS - 1)
            { // q/dO only need to be loaded once for each outer loop
                smem_do.move_to_next_write_buffer();
                gmem_do.move_to(l + 1);
                gmem_do.load(smem_do);
                // smem_q.move_to_next_write_buffer();
                gmem_q.move_to(l + 1);
                gmem_q.load(smem_q);
            }
            else
            {
                // reset the pointer of qq/q/out
                smem_do.move_to_next_write_buffer();
                gmem_do.reset();

                gmem_q.reset();
            }

            // Convert from the accumulator type to FP32 for Softmax.
            softmax.unpack(acc_p);

            float s_mat[2 * M][4 * N]; // to hold P'

#pragma unroll
            for (int mi = 0; mi < M; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < N; ni++)
                {
                    uint4& dst = s_regs[mi][ni];
                    fmha::half2_to_float2(s_mat[2 * mi + 0][4 * ni + 0], s_mat[2 * mi + 0][4 * ni + 1], dst.x);
                    fmha::half2_to_float2(s_mat[2 * mi + 0][4 * ni + 2], s_mat[2 * mi + 0][4 * ni + 3], dst.y);
                    fmha::half2_to_float2(s_mat[2 * mi + 1][4 * ni + 0], s_mat[2 * mi + 1][4 * ni + 1], dst.z);
                    fmha::half2_to_float2(s_mat[2 * mi + 1][4 * ni + 2], s_mat[2 * mi + 1][4 * ni + 3], dst.w);
                }
            }

#pragma unroll
            for (int mi = 0; mi < M; mi++)
            {
#pragma unroll
                for (int ii = 0; ii < 2; ii++)
                {
#pragma unroll
                    for (int ni = 0; ni < N; ni++)
                    {
#pragma unroll
                        for (int jj = 0; jj < 4; jj++)
                        {
                            float& s_dmask = s_mat[2 * mi + ii][4 * ni + jj]; // P'
                            bool const drop = reinterpret_cast<uint32_t const&>(s_dmask) & 0x80000000;
                            // 2. dS = dU * dmask
                            float const d_s
                                = drop ? 0.f : softmax.elt_[2 * mi + ii][4 * ni + jj] * params.rp_dropout; // dP'
                            // 3. tmp = dS * S, dP' * P'
                            s_dmask = fabsf(s_dmask);                                      // P'
                            softmax.elt_[2 * mi + ii][4 * ni + jj] = d_s * fabsf(s_dmask); // dP' * P'
                        }
                    }
                }
            }

            // 4. p_sum = tmp.sum(-1)
            // float p_sum[2 * M];
            // softmax.template reduce<fmha::Sum_>(p_sum);

            // TODO, need to check why scale_bmm1 in softmax is 1
            float const scalef = reinterpret_cast<float const&>(params.scale_softmax);
#pragma unroll
            for (int mi = 0; mi < M; mi++)
            {
#pragma unroll
                for (int ii = 0; ii < 2; ii++)
                {
#pragma unroll
                    for (int ni = 0; ni < N; ni++)
                    {
#pragma unroll
                        for (int jj = 0; jj < 4; jj++)
                        {
                            // 5. dP = (dS - p_sum) * S / sqrtf(d)
                            // dP -= P' * p_sum
                            softmax.elt_[2 * mi + ii][4 * ni + jj]
                                -= softmax_sum[mi * 2 + ii] * (s_mat[2 * mi + ii][4 * ni + jj]);
                            softmax.elt_[2 * mi + ii][4 * ni + jj] *= scalef;
                        }
                    }
                }
            }

            typename Smem_tile_st::Fragment frag_st[Mma_tile_dv::MMAS_K][Mma_tile_dv::MMAS_M]; // P'^T
            smem_st.load(frag_st);                                                             // load P' with transpose
            for (int ki = 0; ki < Mma_tile_dv::MMAS_K; ki++)
            {
                for (int mi = 0; mi < Mma_tile_dv::MMAS_M; mi++)
                {
                    for (int ii = 0; ii < Smem_tile_st::Fragment::NUM_REGS; ii++)
                    {
                        frag_st[ki][mi].reg(ii) = fmha::hmul2(frag_st[ki][mi].reg(ii), params.scale_dropout); // S' ^ T
                        // frag_st[ki][mi].reg(ii) = fmha::hrelu2(frag_st[ki][mi].reg(ii)); // S^T
                    }
                }
            }

            // 5. Store dP
            gmem_s.move(l, newloop); // This function should be called first before used
            gmem_s.store(softmax.elt_, mask);
            // gmem_s.move();

            // 6. Reload Q from smem, but transposed.
            // 7. Accumulate (S * D)' * d_out' for next k-slice
            static_assert(Mma_tile_dv::MMAS_K == 1); // DEBUG
#pragma unroll
            for (int ki = 1; ki < Mma_tile_dv::MMAS_K; ++ki)
            {
                // Trigger the load from shared memory for the next series of Q values.
                smem_dot.load(frag_dot[ki & 1], ki);

                // Do the math for the values already in registers. // dV = S^T * dO
                fmha::gemm(acc_dv, frag_st[(ki - 1)], frag_dot[(ki - 1) & 1]);
            }

            // Do the final stage of math.
            {
                int ki = Mma_tile_dv::MMAS_K;
                fmha::gemm(acc_dv, frag_st[(ki - 1)], frag_dot[(ki - 1) & 1]);
            }
            // Commit the values for Q into shared memory.
            // if( loop + Cta_tile_p::M < Cta_tile_p::N )
            if (l < STEPS - 1)
            {
                gmem_do.commit(smem_do);
                gmem_q.commit(smem_q);
            }

            // Make sure we are reading from the correct buffer.
            smem_do.move_to_next_read_buffer();
            smem_dot.move_to_next_read_buffer();
            // smem_q.move_to_next_read_buffer();

            // Make sure the data is in shared memory.
            fmha::depbar<USE_LDGSTS_Q, 1>();
            __syncthreads();

            // Trigger the loads for the values of Q for the next iteration.
            if (l < STEPS - 1)
            {
                smem_do.load(frag_do[0], 0);
                smem_vt.load(frag_vt[0], 0);
                smem_kt.load(frag_kt[0], 0);
                smem_dot.load(frag_dot[0], 0);
                smem_q.load(frag_q[0], 0);
            }
        } // loop over the sequence length.

        //        if ( newloop < isteps - 1 ) {
        //            gmem_vt.move_to(newloop + 1);
        //            gmem_k.move_to(newloop + 1);
        //        }

        // Epilogue for dV = (S * D)' * d_out'. We're fully exposed to this!

        // Epilogue swizzle for dV
        Smem_tile_dv smem_dv(&smem_[Kernel_traits::Smem_tile_q::BYTES_PER_TILE], tidx);

        /*
    uint4 dv[Mma_tile_dv::MMAS_M][Mma_tile_dv::MMAS_N];
    #pragma unroll
    for( int mi = 0; mi < Mma_tile_dv::MMAS_M; ++mi ) {
        #pragma unroll
        for( int ni = 0; ni < Mma_tile_dv::MMAS_N; ++ni ) {
            // 1st row - 4 elements per row.
            float tmp00 = acc_dv[mi][ni].elt(0);
            float tmp01 = acc_dv[mi][ni].elt(1);
            float tmp02 = acc_dv[mi][ni].elt(4);
            float tmp03 = acc_dv[mi][ni].elt(5);
            // 2nd row - 4 elements per row.
            float tmp10 = acc_dv[mi][ni].elt(2);
            float tmp11 = acc_dv[mi][ni].elt(3);
            float tmp12 = acc_dv[mi][ni].elt(6);
            float tmp13 = acc_dv[mi][ni].elt(7);

            dv[mi][ni].x = fmha::float2_to_half2(tmp00, tmp01);
            dv[mi][ni].y = fmha::float2_to_half2(tmp02, tmp03);
            dv[mi][ni].z = fmha::float2_to_half2(tmp10, tmp11);
            dv[mi][ni].w = fmha::float2_to_half2(tmp12, tmp13);
        }
    }

    smem_dv.store(dv);
    */
        smem_dv.store(acc_dv);
        __syncthreads();
        uint4 dv_out[Smem_tile_dv::NUM_LDS];
        smem_dv.load(dv_out);

        gmem_dv.move_to(newloop);

        gmem_dv.store(dv_out);

        // if ( newloop < isteps - 1 ) {
        //     gmem_dv.move_to(newloop + 1);
        // }

    } // new loop (newloop) over the other sequence length
}

template <typename Kernel_traits, typename Kernel_traits_fp16, typename Params>
inline __device__ void compute_dq_dk_1xN(Params const& params, int const isteps)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dk = typename Traits::template Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M,
        Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;
    // Start of DEBUG
    static_assert(Cta_tile_dk::M == 512 || Cta_tile_dk::M == 384 || Cta_tile_dk::M == 256 || Cta_tile_dk::M == 128
        || Cta_tile_dk::M == 64);
    static_assert(Cta_tile_dk::N == 64 || Cta_tile_dk::N == 128 || Cta_tile_dk::N == 256);
    static_assert(Cta_tile_dk::K == 16);
    // static_assert( Cta_tile_dk::WARPS_M == 4 );
    //  End of DEBUG

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;
    using Mma_tile_o = typename Traits::template Mma_tile<Cta_tile_o>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dk = typename Traits::template Mma_tile<Cta_tile_dk>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_v; // K is used like V in fprop

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Cta_tile_o_fp16 = typename Kernel_traits_fp16::Cta_tile_o;
    using Traits_fp16 = typename Kernel_traits_fp16::Traits_p;
    // using Gmem_tile_dq = typename Kernel_traits::Gmem_tile_o;
    // using Gmem_tile_dq = Gmem_tile_dq<Traits, Cta_tile_o, Kernel_traits::HEADS_INTERLEAVED>;
    using Gmem_tile_dq = Gmem_tile_dq<Traits_fp16, Cta_tile_o_fp16, Kernel_traits_fp16::HEADS_INTERLEAVED>;
    // The shared memory tile to swizzle O.
    // using Smem_tile_dq = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_dq = typename Kernel_traits_fp16::Smem_tile_o;

    // The global memory tile to store dK.
    using Gmem_tile_dk = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dK.
    using Smem_tile_dk = Smem_tile_mma_epilogue<Traits, Cta_tile_dk>;
    static_assert(Smem_tile_dk::NUM_LDS == Gmem_tile_dk::LDGS);
    static_assert(Smem_tile_dk::THREADS_PER_ROW == Gmem_tile_dk::THREADS_PER_ROW);

    // The shared memory tile to reload Q transposed.
    using Smem_tile_qt = fmha::Smem_tile_b<Traits, Cta_tile_dk, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 1>;
    // The shared memory tile to transpose S.
    using Smem_tile_st = Smem_tile_mma_transposed<Traits, Cta_tile_p>; // should be tile P, since it refers to the

    // input tile

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

    enum
    {
        M = Mma_tile_p::MMAS_M
    };

    enum
    {
        N = Mma_tile_p::MMAS_N
    };

    static_assert(M == Mma_tile_o::MMAS_M);
    static_assert(N == Mma_tile_o::MMAS_K);
    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    Block_info_padded<Kernel_traits::THREADS> const binfo(params, bidb, bidh, tidx);
    if (binfo.stop_early())
        return;

    fmha::Mask<Traits, Cta_tile_p, Kernel_traits::VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);
    Smem_tile_qt smem_qt(&smem_[0], tidx);
    Smem_tile_st smem_s(
        &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE + Kernel_traits::Smem_tile_o::BYTES_PER_TILE],
        tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_dq gmem_dq(params, binfo, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_dq smem_dq(&smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], tidx);

    using Gmem_tile_s = Gmem_tile_mma_s<Traits, Cta_tile_p>;
    Gmem_tile_s gmem_s(params, binfo, tidx);

    Qkv_params dk_params;
    dk_params.qkv_ptr = params.dqkv_ptr;
    dk_params.qkv_stride_in_bytes = params.qkv_stride_in_bytes;
    dk_params.h = params.h;
    Gmem_tile_dk gmem_dk(dk_params, 1, binfo, tidx);

    for (int loop_id = 0; loop_id < isteps; loop_id++)
    {
        int newloop = blockIdx.z + loop_id * gridDim.z;
        int const new_loop = newloop * Cta_tile_o::K;
        if (new_loop >= binfo.actual_seqlen)
            // break;
            return;

        // gmem_q.reset(params, binfo);
        // gmem_dq.reset();
        gmem_k.move_to(newloop);

        // Trigger the loads for Q.
        gmem_q.load(smem_q);
        // Trigger the loads for K.
        gmem_k.load(smem_k);

        // 1. Load dP
        uint4 s_regs[M][N];
        gmem_s.move(0 /*q loop*/, newloop /*k loop*/); // This function should be called first before loading
        gmem_s.load(s_regs, mask);
        // gmem_s.move();
        //  Push the LDGDEPBAR instruction after the loads for Q, K and V.
        fmha::ldgdepbar<USE_LDGSTS>();

        // Commit the data for Q and K to shared memory.
        gmem_q.commit(smem_q);
        gmem_k.commit(smem_k);

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS, 1>();
        __syncthreads();

        typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dk::MMAS_N];
        smem_qt.load(frag_qt[0], 0);
#define LOAD_K 0
#if LOAD_K
        // Load the fragments for K. We keep the data in registers during the entire kernel.
        typename Smem_tile_k::Fragment frag_k[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
        for (int ki = 0; ki < Mma_tile_o::MMAS_K; ki++)
        {
            smem_k.load(frag_k[ki], ki);
        }
#else
        typename Smem_tile_k::Fragment frag_k[2][Mma_tile_o::MMAS_N];
        smem_k.load(frag_k[0], 0);
#endif

        enum
        {
            BITS_PER_ELT_S = sizeof(typename Traits::A_type) * 8
        };

        enum
        {
            THREADS_PER_ROW = 32
        };

        int const STEPS = (binfo.actual_seqlen + Cta_tile_p::M - 1) / Cta_tile_p::M;

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator<Traits> acc_dk[Mma_tile_dk::MMAS_M][Mma_tile_dk::MMAS_N];
        fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_dk::WARPS_K>::apply(acc_dk);

        // Load over the entire sequence length.
        // for( int loop = 0; loop < Cta_tile_p::N; loop += Cta_tile_p::M )
        for (int l = 0; l < STEPS; l++)
        {
            int const loop = l * Cta_tile_p::M;
            if (loop >= binfo.actual_seqlen)
                break;

            if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0)
            {
                // if we share K and V,
                //  it could be that V was not fully read yet but we write into smem
                // for reduction
                //__syncthreads();
            }

            // Pack dP as Fragment_a
            fmha::Fragment_a<Traits, fmha::Row> frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
#pragma unroll
            for (int mi = 0; mi < M; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < N; ni++)
                {
                    uint4& dst = s_regs[mi][ni];
                    frag_p[ni][mi].reg(0) = dst.x; // row 0, cols 0,1
                    frag_p[ni][mi].reg(1) = dst.z; // row 8, cols 0,1
                    frag_p[ni][mi].reg(2) = dst.y; // row 0, cols 8,9
                    frag_p[ni][mi].reg(3) = dst.w; // row 8, cols 8,9
                }
            }

            // Declare the accumulators for the 1st gemm.
            fmha::Fragment_accumulator<Traits> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
            fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

            // Do this part of O = P^T * V^T. dQ = dP x dK
#if LOAD_K
#pragma unroll
            for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
            {
                fmha::gemm(acc_o, frag_p[ki], frag_k[ki]); // dQ = dP * K
            }
#else
#pragma unroll
            for (int ki = 1; ki < Mma_tile_o::MMAS_K; ++ki)
            {
                // Trigger the load from shared memory for the next series of Q values.
                smem_k.load(frag_k[ki & 1], ki);
                // Do the math for the values already in registers.
                fmha::gemm(acc_o, frag_p[ki - 1], frag_k[(ki - 1) & 1]);
            }

            // Do the final stage of math.
            {
                int ki = Mma_tile_o::MMAS_K;
                fmha::gemm(acc_o, frag_p[ki - 1], frag_k[(ki - 1) & 1]);
            }
#endif

            // 2. Store dP to smem for transpose
            smem_s.store(s_regs); // dP
            // if( loop + Cta_tile_p::M < Cta_tile_p::N )
            if (l < STEPS - 1)
            {
                // Load next part of S
                gmem_s.move(l + 1 /*q loop*/, newloop /*k loop*/); // This function should be called first before used
                gmem_s.load(s_regs, mask);
                smem_q.move_to_next_write_buffer();
                gmem_q.move_to(l + 1);
                gmem_q.load(smem_q);
            }

// Loop over MMAS_M.
#pragma unroll
            for (int ii = 0; ii < Gmem_tile_dq::LOOPS; ++ii)
            {

                // Swizzle the elements and do the final reduction.
                smem_dq.store(acc_o, ii);

                // Make sure the data is in shared memory.
                __syncthreads();

                // Load from shared memory.
                uint4 out[Gmem_tile_dq::STGS_PER_LOOP];
                smem_dq.load(out);

                // Make sure the data was read from shared memory.
                if (ii < Gmem_tile_dq::LOOPS - 1)
                {
                    __syncthreads();
                }

                // Output the values.
                // if ( newloop == 0) {
                //    gmem_dq.store(out, ii);
                //} else {
                //    // Load from global memory.
                //    uint4 old[Gmem_tile_dq::STGS_PER_LOOP];
                //    gmem_dq.load(old, ii); // TODO should be pipelined?
                //    gmem_dq.store(out, old, ii);
                //}
                gmem_dq.atomic_store(out, ii);
            }

            // Move to the next part of the output.
            if (l < STEPS - 1)
            {
                // gmem_dq.move_to(l + 1); // TODO: this way is incorrect
                gmem_dq.move();
            }

            typename Smem_tile_st::Fragment frag_s[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_M];
            smem_s.load(frag_s); // dP^T ??

            // Trigger the load for the next Q values.
            // if(not_last_iter) {
            //    smem_q.move_to_next_write_buffer();
            //    gmem_q.move();
            //    gmem_q.load( smem_q );
            //}

            // 7. Accumulate (S * D)' * d_out' for next k-slice
            static_assert(Mma_tile_dk::MMAS_K == 1); // DEBUG

#pragma unroll
            for (int ki = 1; ki < Mma_tile_dk::MMAS_K; ++ki)
            {
                // Trigger the load from shared memory for the next series of Q values.
                smem_qt.load(frag_qt[ki & 1], ki); // Q
                // Do the math for the values already in registers.
                fmha::gemm(acc_dk, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]); // dK = dP^T * Q
            }

            // Do the final stage of math.
            {
                int ki = Mma_tile_dk::MMAS_K;
                fmha::gemm(acc_dk, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
            }

            // Commit the values for Q into shared memory.
            // if( loop + Cta_tile_p::M < Cta_tile_p::N )
            if (l < STEPS - 1)
            {
                gmem_q.commit(smem_q);
            }

            // Make sure we are reading from the correct buffer.
            if (USE_LDGSTS_Q)
            {
                smem_q.move_to_next_read_buffer();
            }

            // Make sure the data is in shared memory.
            fmha::depbar<USE_LDGSTS_Q, 1>();
            __syncthreads();

            // Trigger the loads for the values of Q for the next iteration.
            smem_qt.load(frag_qt[0], 0);
            smem_k.load(frag_k[0], 0);

        } // loop over the sequence length.

        gmem_q.reset();
        gmem_dq.reset();
        // gmem_k.move_to(newloop + 1);

        // Epilogue for dK = dP' * dq. We're fully exposed to this!

        // Epilogue swizzle for dK
        Smem_tile_dk smem_dk(&smem_[0], tidx);
        /*
    uint4 dk[Mma_tile_dk::MMAS_M][Mma_tile_dk::MMAS_N];

    #pragma unroll
    for( int mi = 0; mi < Mma_tile_dk::MMAS_M; ++mi ) {
        #pragma unroll
        for( int ni = 0; ni < Mma_tile_dk::MMAS_N; ++ni ) {
            // 1st row - 4 elements per row.
            float tmp00 = acc_dk[mi][ni].elt(0);
            float tmp01 = acc_dk[mi][ni].elt(1);
            float tmp02 = acc_dk[mi][ni].elt(4)//;
            float tmp03 = acc_dk[mi][ni].elt(5)//;
            // 2nd row - 4 elements per row.
            float tmp10 = acc_dk[mi][ni].elt(2);
            float tmp11 = acc_dk[mi][ni].elt(3);
            float tmp12 = acc_dk[mi][ni].elt(6);


            dk[mi][ni].x = fmha::float2_to_half2(tmp00, tmp01);
            dk[mi][ni].y = fmha::float2_to_half2(tmp02, tmp03);
            dk[mi][ni].z = fmha::float2_to_half2(tmp10, tmp11);
            dk[mi][ni].w = fmha::float2_to_half2(tmp12, tmp13);
        }
    }

    smem_dk.store(dk);
    */
        smem_dk.store(acc_dk);
        __syncthreads();
        uint4 dk_out[Smem_tile_dk::NUM_LDS];
        smem_dk.load(dk_out);
        gmem_dk.move_to(newloop);
        gmem_dk.store(dk_out);
        // if ( newloop < isteps - 1 ) {
        //     gmem_dk.move_to(newloop + 1);
        // }
    } // new loop as the outer loop
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
