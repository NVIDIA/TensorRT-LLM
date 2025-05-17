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

// TODO we use this to sneak dqkv ptr into gmem tiles
template <int CHUNKS, typename Kernel_traits, typename Params>
inline __device__ void compute_dv_1xN_nl(Params const& params)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dv = typename Traits::template Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M,
        Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;
    // Start of DEBUG
    static_assert(Cta_tile_dv::M == 512 || Cta_tile_dv::M == 256 || Cta_tile_dv::M == 128);
    static_assert(Cta_tile_dv::N == 64);
    static_assert(Cta_tile_dv::K == 16);
    // static_assert( Cta_tile_dv::WARPS_M == 4 );
    //  End of DEBUG

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dv = typename Traits::template Mma_tile<Cta_tile_dv>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = fmha::Smem_tile_a<Traits, Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;
    // The shared memory tile to reload Q as fragment b.
    using Smem_tile_qt = fmha::Smem_tile_b<Traits, Cta_tile_dv, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

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

    // The global memory tile to store dV.
    using Gmem_tile_dv = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dV.
    using Smem_tile_dv = Smem_tile_mma_epilogue<Traits, Cta_tile_dv>;
    static_assert(Smem_tile_dv::NUM_LDS == Gmem_tile_dv::LDGS);
    static_assert(Smem_tile_dv::THREADS_PER_ROW == Gmem_tile_dv::THREADS_PER_ROW);

    // The shared memory tile to transpose S.
    using Smem_tile_st = Smem_tile_mma_transposed<Traits, Cta_tile_p>; // should be tile P, since it refers to the
                                                                       // input tile
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

    int const bidc = blockIdx.z;
    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    Block_info_padded<Kernel_traits::THREADS> const binfo(params, bidb, bidh, tidx);
    if (binfo.stop_early())
        return;

    using Gmem_tile_do = Gmem_tile_dout<Traits, Cta_tile_p>;

    fmha::Mask<Traits, Cta_tile_p, Kernel_traits::VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_do gmem_q(params, binfo, tidx); // treating d_out as Q
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);
    Smem_tile_qt smem_qt(&smem_[0], tidx);
    Smem_tile_st smem_s(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 2, binfo, tidx); // treating V as K
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Kernel_traits::Smem_tile_q::BYTES_PER_TILE], tidx);

    Gmem_tile_s gmem_s(params.s_ptr, params, tidx);
    using Noloop = Noloop_traits<CHUNKS, Cta_tile_p, 0>;
    Noloop nl_traits(bidc);
    nl_traits.move_all(gmem_q, gmem_s);
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

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
#pragma unroll
    for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
    {
        smem_k.load(frag_k[ki], ki);
    }

    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits::A_type) * 8
    };

    // using Gmem_tile_s = fmha::Gmem_tile_s<Traits, Cta_tile_p, BITS_PER_ELT_S>;
    // Gmem_tile_s gmem_s( params.s_ptr, params.s_stride_in_bytes, params.scale_softmax, tidx );

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE], bidb, tidx);

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

    // Declare the accumulators for the 2nd gemm.
    fmha::Fragment_accumulator<Traits> acc_dv[Mma_tile_dv::MMAS_M][Mma_tile_dv::MMAS_N];
    fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_dv::WARPS_K>::apply(acc_dv);

    // Load over the entire sequence length.
    // for( int loop = 0, outer = 0; loop < Cta_tile_p::N; loop += Cta_tile_p::M, outer++ ) {
    for (int l = 0; l < Noloop::NUM_STEPS; l++)
    {
        int const loop = nl_traits.offset_loop_count(l);
        if (loop >= binfo.actual_seqlen)
            break;

        // 1. Load S
        uint4 s_regs[M][N];
        gmem_s.load(s_regs, mask);
        // 2. Store s * dmask to smem for transpose
        smem_s.store(s_regs);
        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);
// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the next series of Q values.
            smem_q.load(frag_q[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack(acc_p);

        __syncthreads();

        float s_mat[2 * M][4 * N];

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

        float d_s[2 * M][4 * N];

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
                        float const s_dmask = s_mat[2 * mi + ii][4 * ni + jj];
                        bool const drop = reinterpret_cast<uint32_t const&>(s_dmask) & 0x80000000;
                        // 2. dS = dU * dmask
                        d_s[2 * mi + ii][4 * ni + jj] = drop ? 0.f : softmax.elt_[2 * mi + ii][4 * ni + jj];
                        // 3. tmp = dS * S
                        softmax.elt_[2 * mi + ii][4 * ni + jj] = d_s[2 * mi + ii][4 * ni + jj] * fabsf(s_dmask);
                    }
                }
            }
        }

        // 4. p_sum = tmp.sum(-1)
        float p_sum[2 * M];
        softmax.template reduce<fmha::Sum_>(p_sum);

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
                        float const scalef = reinterpret_cast<float const&>(params.scale_softmax);
                        softmax.elt_[2 * mi + ii][4 * ni + jj] = (d_s[2 * mi + ii][4 * ni + jj] - p_sum[2 * mi + ii])
                            * fabsf(s_mat[2 * mi + ii][4 * ni + jj]) * scalef;
                    }
                }
            }
        }
        // Trigger the load for the next Q values. We're using double buffering, so reading qt is safe
        if (loop + Cta_tile_p::M < Cta_tile_p::N)
        {
            smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
        }

        typename Smem_tile_st::Fragment frag_s[Mma_tile_dv::MMAS_K][Mma_tile_dv::MMAS_M];
        smem_s.load(frag_s);
        for (int ki = 0; ki < Mma_tile_dv::MMAS_K; ki++)
        {
            for (int mi = 0; mi < Mma_tile_dv::MMAS_M; mi++)
            {
                for (int ii = 0; ii < Smem_tile_st::Fragment::NUM_REGS; ii++)
                {
                    frag_s[ki][mi].reg(ii) = fmha::hmul2(frag_s[ki][mi].reg(ii), params.scale_dropout);
                    frag_s[ki][mi].reg(ii) = fmha::hrelu2(frag_s[ki][mi].reg(ii));
                }
            }
        }

        // 5. Store dP
        gmem_s.store(softmax.elt_, mask);
        gmem_s.move();

        // 6. Reload Q from smem, but transposed.
        typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dv::MMAS_N];
        static_assert(Smem_tile_qt::Fragment::NUM_REGS == 4);
        static_assert(Mma_tile_dv::MMAS_K == 1);
        smem_qt.load(frag_qt[0], 0);

        // 7. Accumulate (S * D)' * d_out' for next k-slice
        static_assert(Mma_tile_dv::MMAS_K == 1); // DEBUG
#pragma unroll
        for (int ki = 1; ki < Mma_tile_dv::MMAS_K; ++ki)
        {
            assert(false);
            // Trigger the load from shared memory for the next series of Q values.
            smem_qt.load(frag_qt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dv, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dv::MMAS_K;
            fmha::gemm(acc_dv, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Commit the values for Q into shared memory.
        if (loop + Cta_tile_p::M < Cta_tile_p::N)
        {
            gmem_q.commit(smem_q);
        }

        // Make sure we are reading from the correct buffer.
        // if( USE_LDGSTS_Q ) {
        //    smem_q.move_to_next_read_buffer();
        //}
        smem_q.move_to_next_read_buffer();
        smem_qt.move_to_next_read_buffer();

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS_Q, 1>();
        __syncthreads();

        // Trigger the loads for the values of Q for the next iteration.
        smem_q.load(frag_q[0], 0);

    } // Outer loop over the sequence length.

    // Epilogue for dV = (S * D)' * d_out'. We're fully exposed to this!

    // Epilogue swizzle for dV
    Smem_tile_dv smem_dv(&smem_[Kernel_traits::Smem_tile_q::BYTES_PER_TILE], tidx);
    smem_dv.store(acc_dv);

    __syncthreads();

    uint4 dv_out[Smem_tile_dv::NUM_LDS];
    smem_dv.load(dv_out);
    Qkv_params dv_params;
    dv_params.qkv_ptr = params.dkv_ptr;
    dv_params.qkv_stride_in_bytes = params.h * 2 * CHUNKS * params.d * sizeof(half);
    dv_params.h = params.h;
    Gmem_tile_dv gmem_dv(dv_params, nl_traits.get_idx_dv(), binfo, tidx);
    gmem_dv.store(dv_out);
}

template <int CHUNKS, typename Kernel_traits, typename Params>
inline __device__ void compute_dq_dk_1xN_nl(Params const& params)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dk = typename Traits::template Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M,
        Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;
    // Start of DEBUG
    static_assert(Cta_tile_dk::M == 512 || Cta_tile_dk::M == 256 || Cta_tile_dk::M == 128);
    static_assert(Cta_tile_dk::N == 64);
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
    // using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_q = fmha::Smem_tile_a<Traits, Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_v; // K is used like V in fprop

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = Gmem_tile_dq<Traits, Cta_tile_o, Kernel_traits::HEADS_INTERLEAVED>;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    // The global memory tile to store dK.
    using Gmem_tile_dk = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dK.
    using Smem_tile_dk = Smem_tile_mma_epilogue<Traits, Cta_tile_dk>;
    static_assert(Smem_tile_dk::NUM_LDS == Gmem_tile_dk::LDGS);
    static_assert(Smem_tile_dk::THREADS_PER_ROW == Gmem_tile_dk::THREADS_PER_ROW);

    // The shared memory tile to reload Q transposed.
    using Smem_tile_qt = fmha::Smem_tile_b<Traits, Cta_tile_dk, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;
    // The shared memory tile to transpose S.
    using Smem_tile_st = Smem_tile_mma_transposed<Traits, Cta_tile_p>; // should be tile P, since it refers to the
                                                                       // input tile
    using Gmem_tile_s = Gmem_tile_mma_s<Traits, Cta_tile_p>;

    using Noloop = Noloop_traits<CHUNKS, Cta_tile_p, 0>;

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

    int const bidc = blockIdx.z;
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

    Gmem_tile_s gmem_s(params.s_ptr, params, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);
    Smem_tile_qt smem_qt(&smem_[0], tidx);
    Smem_tile_st smem_s(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Kernel_traits::Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    Noloop nl_traits(bidc);

    nl_traits.move_all(gmem_q, gmem_o, gmem_s);
    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);

    // 1. Load dP
    uint4 s_regs[M][N];
    gmem_s.load(s_regs, mask);
    gmem_s.move();
    // Push the LDGDEPBAR instruction after the loads for Q, K and V.
    fmha::ldgdepbar<USE_LDGSTS>();

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_k.commit(smem_k);

    // Make sure the data is in shared memory.
    fmha::depbar<USE_LDGSTS, 1>();
    __syncthreads();

    typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dk::MMAS_N];
    smem_qt.load(frag_qt[0], 0);

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
#pragma unroll
    for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
    {
        smem_k.load(frag_k[ki], ki);
    }

    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits::A_type) * 8
    };

    enum
    {
        THREADS_PER_ROW = 32
    };

    // Declare the accumulators for the 2nd gemm.
    fmha::Fragment_accumulator<Traits> acc_dk[Mma_tile_dk::MMAS_M][Mma_tile_dk::MMAS_N];
    fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_dk::WARPS_K>::apply(acc_dk);

    // Load over the entire sequence length.
    // for( int loop = 0, outer = 0; loop < Cta_tile_p::N; loop += Cta_tile_p::M, outer++ ) {
    for (int l = 0; l < Noloop::NUM_STEPS; l++)
    {
        int const loop = nl_traits.offset_loop_count(l);
        if (loop >= binfo.actual_seqlen)
            break;

        if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V && loop == 0)
        {
            // if we share K and V, it could be that V was not fully read yet but we write into smem
            // for reduction
            __syncthreads();
        }

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

// Do this part of O = P^T * V^T.
#pragma unroll
        for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
        {
            fmha::gemm(acc_o, frag_p[ki], frag_k[ki]);
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
            // if( ii < Gmem_tile_o::LOOPS - 1 ) {
            __syncthreads();
            //}

            // Output the values.
            gmem_o.store(out, ii);
        }

        // Move to the next part of the output.
        gmem_o.move();

        // 2. Store dP to smem for transpose
        smem_s.store(s_regs);

        gmem_s.load(s_regs, mask);
        gmem_s.move();
        __syncthreads();

        typename Smem_tile_st::Fragment frag_s[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_M];
        smem_s.load(frag_s);

        __syncthreads();
        // 6. load Q from smem, but transposed.

        // 7. Accumulate (S * D)' * d_out' for next k-slice
        static_assert(Mma_tile_dk::MMAS_K == 1); // DEBUG
        // Trigger the load for the next Q values.
        if (loop + Cta_tile_p::M < Cta_tile_p::N)
        {
            smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
        }

#pragma unroll
        for (int ki = 1; ki < Mma_tile_dk::MMAS_K; ++ki)
        {
            assert(false);
            // Trigger the load from shared memory for the next series of Q values.
            smem_qt.load(frag_qt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dk, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dk::MMAS_K;
            fmha::gemm(acc_dk, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Commit the values for Q into shared memory.
        if (loop + Cta_tile_p::M < Cta_tile_p::N)
        {
            gmem_q.commit(smem_q);
        }

        // Make sure we are reading from the correct buffer.
        if (USE_LDGSTS_Q)
        {
            // smem_q.move_to_next_read_buffer();
        }
        smem_q.move_to_next_read_buffer();
        smem_qt.move_to_next_read_buffer();

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS_Q, 1>();
        __syncthreads();

        // Trigger the loads for the values of Q for the next iteration.
        smem_qt.load(frag_qt[0], 0);

    } // Outer loop over the sequence length.

    // Epilogue for dK = dP' * dq. We're fully exposed to this!

    // Epilogue swizzle for dK
    Smem_tile_dk smem_dk(&smem_[0], tidx);
    smem_dk.store(acc_dk);

    __syncthreads();

    uint4 dk_out[Smem_tile_dk::NUM_LDS];
    smem_dk.load(dk_out);
    Qkv_params dk_params;
    dk_params.qkv_ptr = params.dkv_ptr;
    dk_params.qkv_stride_in_bytes = params.h * 2 * CHUNKS * params.d * sizeof(half);
    dk_params.h = params.h;
    Gmem_tile_dk gmem_dk(dk_params, nl_traits.get_idx_dk(), binfo, tidx);
    gmem_dk.store(dk_out);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
