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
template <typename Kernel_traits>
struct Gemm_Q_K_base
{
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Fragment_q = typename Smem_tile_q::Fragment;
    using Fragment_k = typename Smem_tile_k::Fragment;
    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = typename Traits::template Mma_tile<Cta_tile_p>;

    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    __device__ inline Gemm_Q_K_base(char* smem_ptr_q, char* smem_ptr_k, int const tidx)
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx)
    {
    }

    __device__ inline void load_q()
    {
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K; ki++)
        {
            smem_q.load(frag_q[ki], ki);
        }
    }

    Fragment_q frag_q[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool K_in_regs>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits>
{

    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;

    enum
    {
        SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V
    };

    enum
    {
        SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE
    };

    enum
    {
        SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE)
    };

    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
        + std::max((SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE,
            Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX);

    __device__ inline Gemm_Q_K(char* smem_, int const tidx)
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx)
    {
    }

    __device__ inline void load_k()
    {
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            Base::smem_k.load(frag_k[ki], ki);
        }
    }

    template <typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N])
    {
// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            // Do the math for the values already in registers.
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_p, Base::frag_q[(ki - 1)], frag_k[(ki - 1)]);
            }
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_p, Base::frag_q[(ki - 1)], frag_k[(ki - 1)]);
            }
        }
    }

    __device__ inline void reload_k()
    {
        // Noop.
    }

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
struct Gemm_Q_K<Kernel_traits, false> : public Gemm_Q_K_base<Kernel_traits>
{
    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;

    Fragment_k frag_k[2][Mma_tile_p::MMAS_N];

    enum
    {
        SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V
    };

    enum
    {
        SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE)
    };

    static_assert(Smem_tile_v::BYTES_PER_TILE == (int) Smem_tile_k::BYTES_PER_TILE);

    enum
    {
        SMEM_OFFSET_O = SMEM_OFFSET_V + Smem_tile_v::BYTES_PER_TILE
    };

    // Q | K/V + O + SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
        + (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE
        + Base::SMEM_BYTES_SOFTMAX;

    __device__ inline Gemm_Q_K(char* smem_, int const tidx)
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx)
    {
    }

    __device__ inline void load_k()
    {
        Base::smem_k.load(frag_k[0], 0);
    }

    template <typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N])
    {
// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
        for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_p, Base::frag_q[(ki - 1)], frag_k[(ki - 1) & 1]);
            }
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            if (ki <= Mma_tile_p::VALID_MMAS_K)
            {
                fmha::gemm(acc_p, Base::frag_q[(ki - 1)], frag_k[(ki - 1) & 1]);
            }
        }
    }

    __device__ inline void reload_k()
    {
        Base::smem_k.load(frag_k[0], 0);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
constexpr size_t get_dynamic_smem_size()
{
    return Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>::SMEM_BYTES;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool IS_TRAINING, typename Params, typename Prng>
inline __device__ void device_flash_attention_(
    Params const& params, int const bidb, int const bidh, int const begin, int const steps, Prng& ph)
{

    // The instruction traits.
    using Traits = typename Kernel_traits::Traits_p;

    // The element type (fp16 or bf16 on A100).
    using Data_type = typename Traits::A_type;

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

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = Gmem_tile_mma_s<Traits, Cta_tile_p>;

    using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>;

    using Softmax = fmha::Softmax<Traits, Cta_tile_p, Kernel_traits>;

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

    // The number of threads per row.
    enum
    {
        THREADS_PER_ROW = 32
    };

    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits::A_type) * 8
    };

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    int const tidx = threadIdx.x;
    // q_loops is configured as blockIdx.y.
    // int q_loop = blockIdx.y;

    Block_info_padded<Kernel_traits::THREADS, Kernel_traits::SEQUENCES_INTERLEAVED> const binfo(
        params, bidb, bidh, tidx);
    if (binfo.stop_early())
        return;

    Gemm1 gemm_q_k(smem_, tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);
    // Wind gmem tiles to the correct position.

    fmha::Mask<Traits, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);
    // Determine whether causal mask is applied or not
    bool const causal_mask = Kernel_traits::MASK_VERSION == 3;

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params, 2, binfo, tidx);
    // The base pointer of smem_v;
    char* smem_v_ = &smem_[Gemm1::SMEM_OFFSET_V];

    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Gemm1::SMEM_OFFSET_O], tidx);

    // Create the object to do the softmax.
    Softmax softmax(params, &smem_[Gemm1::SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE], bidb, tidx);

    // const int kv_loop_bound =
    //     ((binfo.actual_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N);

    // Loop over the local sequence length.
    for (int q_loop = 0; q_loop < steps; q_loop++)
    {
        if ((begin + q_loop) * Cta_tile_p::M >= binfo.actual_seqlen)
        {
            return;
        }

        // Move to the correct location.
        gmem_q.move_to(begin + q_loop);
        // Trigger the loads for Q.
        gmem_q.load(gemm_q_k.smem_q);
        // Trigger the loads for K.
        gmem_k.load(gemm_q_k.smem_k);
        // Trigger the loads for V.
        gmem_v.load(smem_v);

        static_assert(!USE_LDGSTS_K);

        // Push the LDGDEPBAR instruction after the loads for Q, K and V.
        fmha::ldgdepbar<USE_LDGSTS>();

        // Commit the data for Q and K to shared memory.
        gmem_q.commit(gemm_q_k.smem_q);

        // Commit the data for K to shared memory.
        gmem_v.commit(smem_v);

        // Commit the data for K to shared memory.
        if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
        {
            gmem_k.commit(gemm_q_k.smem_k);
        }

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS, 1>();
        __syncthreads();

        // Load the fragments for Q.
        gemm_q_k.load_q();

        // Load the fragments for V. We keep the data in registers during the entire kernel.
        typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::VALID_MMAS_N];
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
            gmem_k.commit(gemm_q_k.smem_k);

            // Make sure the data is in shared memory.
            __syncthreads();
        }

        // Load the fragments for K.
        gemm_q_k.load_k();

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
        using Acc_type_o = typename Traits::Accumulator_type;
        fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

        // Flash attention updater
        fmha::Fragment_updater<Traits, Cta_tile_o> acc_o_updater(params, binfo);

        // Load the mask for that iteration.
        mask.load(begin + q_loop);

        // avoid unnecessary computation if causal mask is applied
        static_assert(Cta_tile_p::M % Cta_tile_p::N == 0);
        int kv_loop_bound = causal_mask ? ((begin + q_loop + 1) * Cta_tile_p::M) / Cta_tile_p::N
                                        : ((binfo.actual_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N);

        for (int kv_loop = 0; kv_loop < kv_loop_bound; kv_loop++)
        {

            // Declare the accumulators for the 1st gemm.
            fmha::Fragment_accumulator<Traits> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            fmha::Clear_accumulator<typename Traits::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

            // Do this part of P^T = (Q * K^T)^T.
            gemm_q_k(acc_p);

            if (kv_loop + 1 < kv_loop_bound)
            {
                gmem_k.move_to(kv_loop + 1);
                gemm_q_k.smem_k.move_to_next_write_buffer();
                gmem_v.move_to(kv_loop + 1);
                smem_v.move_to_next_write_buffer();
            }

            // reset kv gmem offset for next q loop
            if (kv_loop + 1 >= kv_loop_bound)
            {
                gmem_k.reset();
                gmem_v.reset();
            }

            if (kv_loop + 1 < kv_loop_bound)
            {
                gmem_k.load(gemm_q_k.smem_k);
                gmem_v.load(smem_v);
            }

            // Push the LDGDEPBAR instruction after the loads for K.
            fmha::ldgdepbar<USE_LDGSTS>();

            // Convert from the accumulator type to FP32 for Softmax.
            softmax.unpack(acc_p);

            mask.load(begin + q_loop, kv_loop);

            // Apply the mask.
            if (params.has_alibi)
            {
                softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
            }
            else
            {
                softmax.apply_mask(mask);
            }

            // check whether it is needed
            if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
            {
                // if we share K and V, it could be that V was not fully read yet but
                //  we write into smem for reduction
                __syncthreads();
            }

            // Compute the max.
            // float p_max[Mma_tile_p::MMAS_M * 2];
            // softmax.template reduce<fmha::Max_>(p_max);
            // softmax.reduce_max(acc_o_updater.p_max);
            softmax.template reduce<fmha::Max_>(acc_o_updater.prev_max_);

            // Update acc_max scale of flash attention
            acc_o_updater.update_acc_max();

            // Compute the exponential value.
            softmax.apply_exp(acc_o_updater.curr_max_);

            // Compute the sum.
            // float p_sum[Mma_tile_p::MMAS_M * 2];
            // softmax.reduce_sum(acc_o_updater.p_sum);
            softmax.template reduce<fmha::Sum_>(acc_o_updater.prev_sum_);

            // Update acc_sum scale of flash attention
            acc_o_updater.update_acc_sum();

            // Finalize softmax on the accumulators of P^T.
            // NOTE: for training forward, we need to scale p before feeding into P * V
            softmax.scale(acc_o_updater.curr_sum_);

            using Frag_p = fmha::Fragment_a<Traits, fmha::Row>;
            Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
            if (IS_TRAINING)
            {
                if (params.has_dropout)
                {
                    // for warp Nx1
                    static_assert(Cta_tile_p::WARPS_N == 1);

                    // 32 means the number of thread per warp
                    unsigned int warp_idx = threadIdx.x / 32;
                    // 16 means each warp will handle 16 x 16 elements in current MMA computation.
                    unsigned int block_col_idx = kv_loop * Cta_tile_p::N / 16;
                    unsigned long long philox_subsequence
                        = ((begin + q_loop) * Cta_tile_p::M / 16 + warp_idx) * (binfo.actual_seqlen / 16)
                        + block_col_idx;

#pragma unroll
                    for (int mi = 0; mi < Mma_tile_p::MMAS_M; mi++)
                    {
#pragma unroll
                        for (int ni = 0; ni < Mma_tile_p::MMAS_N; ni++)
                        {

                            // convert 4 int32_t to 8 int16_t
                            uint16_t* rand_arr = (uint16_t*) &ph(philox_subsequence + ni * Cta_tile_p::WARPS_N
                                + mi * Cta_tile_p::WARPS_M * (binfo.actual_seqlen / 16));

#pragma unroll
                            for (int jj = 0; jj < 2; jj++)
                            {
#pragma unroll
                                for (int ii = 0; ii < 4; ii++)
                                {
                                    // We encode the dropout pattern in the sign bit of
                                    //  the non-negative softmax to distinguish from pre-existing zeros
                                    softmax.elt_[2 * mi + jj][4 * ni + ii]
                                        = rand_arr[4 * jj + ii] <= params.p_dropout_16bit
                                        ? softmax.elt_[2 * mi + jj][4 * ni + ii]
                                        : -softmax.elt_[2 * mi + jj][4 * ni + ii];
                                } // ii
                            }     // jj
                        }         // ni
                    }             // mi
                }                 // has_dropout

                softmax.pack(frag_p);
                // for debug, store the sign of dropout for testing
                gmem_s.move(begin + q_loop, kv_loop); // Note the second param is kv_loop_id
                gmem_s.store_sign(softmax.elt_, mask);
            }
            else
            {
                softmax.pack(frag_p);
            }

            // make sure packing frag_p is done
            __syncthreads();

            // can only be committed after softmax as they share the smem
            if (kv_loop + 1 < kv_loop_bound)
            {
                gmem_v.commit(smem_v);
            }
            // Commit the data for K to shared memory.
            if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V && kv_loop + 1 < kv_loop_bound)
            {
                gmem_k.commit(gemm_q_k.smem_k);
            }

            if (params.has_dropout)
            {
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
                            frag_p[ki][mi].reg(ii)
                                = fmha::mul2<Data_type>(frag_p[ki][mi].reg(ii), params.scale_dropout);
                            frag_p[ki][mi].reg(ii) = fmha::relu2<Data_type>(frag_p[ki][mi].reg(ii));
                        }
                    }
                }
            }

            // Update acc_o of flash attention (save registers)
            acc_o_updater.update_o(acc_o);

            // Make sure we have the LDGDEPBAR in place.
            fmha::ldgdepbar<USE_LDGSTS_Q>();

// Do this part of O = P^T * V^T.
#pragma unroll
            for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
            {
                fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
            }

            fmha::depbar<USE_LDGSTS, 1>();
            __syncthreads();

// Load the fragments for next K.
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
                gmem_k.commit(gemm_q_k.smem_k);

                // Make sure the data is in shared memory.
                __syncthreads();
            }

            // Load the fragments for K.
            gemm_q_k.load_k();

            // need to reload q as Q_IN_REGS=false by default
            if (q_loop + 1 < steps && kv_loop + 1 == kv_loop_bound)
            {
                gemm_q_k.load_q();
            }

        } // kv_loop

        // store softmax (m + log(l)) to global memory
        acc_o_updater.store(begin + q_loop);

        gmem_o.move_to(begin + q_loop);
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
        gemm_q_k.reload_k();

    } // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool IS_TRAINING, typename Params>
inline __device__ void device_flash_attention(Params const& params, int const num_full_heads, int const num_main_groups,
    int const main_group_size, int const main_steps, int const rest_steps)
{

    int const STEPS = (params.s + Kernel_traits::Cta_tile_p::M - 1) / Kernel_traits::Cta_tile_p::M;
    int const tidx_global = blockIdx.x * gridDim.x + threadIdx.x;
    auto seeds = at::cuda::philox::unpack(params.philox_args);

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        uint64_t* ptr = reinterpret_cast<uint64_t*>(params.seed_ptr);
        ptr[0] = std::get<0>(seeds);
        ptr[1] = std::get<1>(seeds);
    }

    for (int it = 0; it < num_full_heads; it++)
    {
        int const bidx = it * gridDim.x + blockIdx.x;
        int const bidh = bidx % params.h;
        int const bidb = bidx / params.h;
        Philox ph(std::get<0>(seeds), 0, std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + threadIdx.x % 32);
        fused_multihead_attention::device_flash_attention_<Kernel_traits, IS_TRAINING>(
            params, bidb, bidh, 0, STEPS, ph);
        __syncthreads();
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
        Philox ph(std::get<0>(seeds), 0, std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + threadIdx.x % 32);
        fused_multihead_attention::device_flash_attention_<Kernel_traits, IS_TRAINING>(
            params, bidb, bidh, offset, main_steps, ph);
    }
    else
    {
        if (rest_steps == 0)
            return;
        // process across heads
        int const bidx = blockIdx.x - main_group_size * num_main_groups;
        int const offset = num_main_groups * main_steps;
        int const total_heads = params.b * params.h;
        int const rest_ctas = gridDim.x - main_group_size * num_main_groups;
        for (int it = head_offset + bidx; it < total_heads; it += rest_ctas)
        {
            int const bidh = it % params.h;
            int const bidb = it / params.h;
            Philox ph(std::get<0>(seeds), 0, std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + threadIdx.x % 32);
            fused_multihead_attention::device_flash_attention_<Kernel_traits, IS_TRAINING>(
                params, bidb, bidh, offset, rest_steps, ph);
            __syncthreads();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool IS_TRAINING, typename Params>
inline __device__ void device_flash_attention(Params const& params, int const total_heads)
{

    // std::make_tuple(arg.seed_.val, arg.offset_.val);
    // https://github.com/pytorch/pytorch/blob/
    //  fc94a2115b31dfe7a0d8f28eb4f5ed532c4f0792/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L18-L28
    auto seeds = at::cuda::philox::unpack(params.philox_args);

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        uint64_t* ptr = reinterpret_cast<uint64_t*>(params.seed_ptr);
        ptr[0] = std::get<0>(seeds);
        ptr[1] = std::get<1>(seeds);
    }
    int const STEPS = (params.s + Kernel_traits::Cta_tile_p::M - 1) / Kernel_traits::Cta_tile_p::M;

    for (int bidx = blockIdx.x; bidx < total_heads; bidx += gridDim.x)
    {
        int const bidh = bidx % params.h;
        int const bidb = bidx / params.h;
        int const tidx = threadIdx.x;
        Philox ph(std::get<0>(seeds), 0, std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32);

        fused_multihead_attention::device_flash_attention_<Kernel_traits, IS_TRAINING>(
            params, bidb, bidh, 0, STEPS, ph);
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
