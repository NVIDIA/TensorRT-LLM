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
inline __device__ void device_flash_attention(Params const& params)
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
        USE_LDGSTS_V = Kernel_traits::USE_LDGSTS_V
    };

    enum
    {
        USE_LDGSTS_KV = Kernel_traits::USE_LDGSTS_K || Kernel_traits::USE_LDGSTS_V
    };

    // Do we use LDGSTS for any of the 3 input matrices.
    enum
    {
        USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V
    };

    // If either K or V uses LDGSTS, they cannot share a buffer.
    static_assert(!(USE_LDGSTS_K || USE_LDGSTS_V) || !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");

    // Fragment double buffer (reduce register pressure)
    enum
    {
        FRAGMENT_K_SIZE_IN_K_DIM
        = (Kernel_traits::LIMIT_QK_FRAGMENTS) *2 + !(Kernel_traits::LIMIT_QK_FRAGMENTS) *Mma_tile_p::MMAS_K
    };

    static_assert(!(Kernel_traits::LIMIT_QK_FRAGMENTS && USE_LDGSTS_K), "");

    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the batch.
    int const bidb = blockIdx.y;
    // The block index for the head.
    int const bidh = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    // The block info.
    Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
    if (binfo.stop_early())
    {
        return;
    }

    // Create the object to control the masks.
    fmha::Mask<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    // Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
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

    // Allocate the global memory tile loader for O.
    // Gmem_tile_o gmem_o(params, binfo, tidx);
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

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
    gmem_k.commit(smem_k);

    // Commit the data for V to shared memory.
    if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
    {
        gmem_v.commit(smem_v);
    }

    // Make sure the data is in shared memory.
    fmha::depbar<USE_LDGSTS, 1>();
    __syncthreads();

    // Load the fragments for Q.
    // NOTE: lds full frag_q as it will be used by all kv loops
    typename Smem_tile_q::Fragment frag_q[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
#pragma unroll
    for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
    {
        smem_q.load(frag_q[ki], ki);
    }

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[FRAGMENT_K_SIZE_IN_K_DIM][Mma_tile_p::MMAS_N];
    if (Kernel_traits::LIMIT_QK_FRAGMENTS)
    {
        smem_k.load(frag_k[0], 0);
    }
    else
    {
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            smem_k.load(frag_k[ki], ki);
        }
    }

    // Commit the data for V to shared memory if it has not been done already.
    if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
    {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_v.commit(smem_v);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::VALID_MMAS_N];
#pragma unroll
    for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
    {
        smem_v.load(frag_v[ki], ki);
    }

    // Store/load P to/from memory (for debugging).
#if defined(STORE_P)
    enum
    {
        BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8
    };

    using Gmem_tile_p = fmha::Gmem_tile_p<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
    Gmem_tile_p gmem_p(params.p_ptr, params.p_stride_in_bytes, params.scale_bmm1, tidx);
#endif

    // Store S to memory (for debugging). NOTE: We use A_type as C_type is int32 for IMMA???
#if defined(STORE_S)
    enum
    {
        BITS_PER_ELT_S = sizeof(typename Traits_p::A_type) * 8
    };

    using Gmem_tile_s = fmha::Gmem_tile_s<Traits_p, Cta_tile_p, BITS_PER_ELT_S>;
    Gmem_tile_s gmem_s(params.s_ptr, params.s_stride_in_bytes, params.scale_softmax, tidx);
#endif

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
    Softmax softmax(params, &smem_[Smem_tile_q::BYTES_PER_TILE], bidb, tidx);

    // Prefetch next kv buffer to share memory
    enum
    {
        PREFETCH_K_BUFFER_TO_SMEM = !Kernel_traits::LIMIT_QK_FRAGMENTS && !Softmax::USE_SHARED_MEMORY
    };

    enum
    {
        PREFETCH_V_BUFFER_TO_SMEM = !Kernel_traits::SHARE_SMEM_FOR_K_AND_V
    };

    enum
    {
        PREFETCH_KV_BUFFER_TO_SMEM = PREFETCH_K_BUFFER_TO_SMEM && PREFETCH_V_BUFFER_TO_SMEM
    };

    // The number of threads per row.
    // enum { THREADS_PER_ROW = Cta_tile_p::WARPS_N * 8 };
    // DEBUG.
    // static_assert(THREADS_PER_ROW == 32, "");
    // END OF DEBUG.
    enum
    {
        THREADS_PER_ROW = 32
    };

    int const q_loop_bound = ((binfo.actual_seqlen + Cta_tile_p::M - 1) / Cta_tile_p::M) * Cta_tile_p::M;

    int const kv_loop_bound = ((binfo.actual_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N) * Cta_tile_p::N;
    // Load over the entire sequence length.
    for (int q_loop = 0, outer = 0; q_loop < q_loop_bound; q_loop += Cta_tile_p::M, outer++)
    {

        // Reset the mask col.
        mask.reset();
        // Load the mask row.
        mask.load(outer);

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
        using Acc_type_o = typename Traits_o::Accumulator_type;
        fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

        // Flash attention updater
        fmha::Fragment_updater<Traits_o, Cta_tile_o> acc_o_updater;

        for (int kv_loop = 0; kv_loop < kv_loop_bound; kv_loop += Cta_tile_p::N)
        {

            // NOTE: causal mask
            if (Kernel_traits::MASK_VERSION == 3 && q_loop + Cta_tile_p::M <= kv_loop)
            {
                break;
            }

            // Trigger the load for the next K/V values (smem_k double buffer).
            if (kv_loop + Cta_tile_p::N < kv_loop_bound && PREFETCH_KV_BUFFER_TO_SMEM)
            {

                // Make sure we are done reading the data (smem).
                if (!(Smem_tile_k::BUFFERS_PER_TILE > 1 && Smem_tile_v::BUFFERS_PER_TILE > 1))
                {
                    __syncthreads();
                }

                gmem_k.move();
                smem_k.move_to_next_write_buffer();
                gmem_k.load(smem_k);

                gmem_v.move();
                smem_v.move_to_next_write_buffer();
                gmem_v.load(smem_v);

                // Push the LDGDEPBAR instruction after the loads for K.
                fmha::ldgdepbar<USE_LDGSTS>();

                gmem_k.commit(smem_k);
                if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
                {
                    gmem_v.commit(smem_v);
                }
            }

            // Declare the accumulators for the 1st gemm.
            fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            using Acc_type_p = typename Traits_p::Accumulator_type;
            fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
            for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
            {

                int k_ki = (ki - 1);
                if (Kernel_traits::LIMIT_QK_FRAGMENTS)
                {
                    k_ki = (ki - 1) & 1;
                    smem_k.load(frag_k[(ki & 1)], ki);
                }

                // Do the math for the values already in registers.
                if (ki <= Mma_tile_p::VALID_MMAS_K)
                {
                    fmha::gemm(acc_p, frag_q[(ki - 1)], frag_k[k_ki]);
                }
            }

            if (Mma_tile_p::MMAS_K <= Mma_tile_p::VALID_MMAS_K)
            {
                int ki = Mma_tile_p::MMAS_K;
                int k_ki = (ki - 1);
                if (Kernel_traits::LIMIT_QK_FRAGMENTS)
                {
                    k_ki = (ki - 1) & 1;
                }
                fmha::gemm(acc_p, frag_q[(ki - 1)], frag_k[k_ki]);
            }

            // Store the P matrix.
#if defined(STORE_P)
            gmem_p.store(acc_p);
#endif

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

            // Move mask to next kv tile
            mask.move();

            // Make sure we are done reading the data (smem_v).
            if ((Kernel_traits::SHARE_SMEM_FOR_K_AND_V || Kernel_traits::LIMIT_QK_FRAGMENTS)
                && Softmax::USE_SHARED_MEMORY)
            {
                __syncthreads();
            }

            // Enable our trick to use the max for INT8 to scale.
            // float p_max[Softmax::ROWS_PER_THREAD];
            if (Kernel_traits::USE_SCALE_MAX)
            {
                // 16129 == 127 ^ 2.
                float p_max = reinterpret_cast<float const&>(params.scale_bmm1) * 16129.f;
                softmax.apply_exp(p_max);
            }
            else
            {
                // Compute the max.
                softmax.template reduce<fmha::Max_>(acc_o_updater.prev_max_);

                // Update acc_max scale of flash attention
                acc_o_updater.update_acc_max();

                if (Softmax::USE_SHARED_MEMORY)
                {
                    // Make sure we are done reading shared memory.
                    __syncthreads();
                }

                // Compute the exponential value.
                softmax.apply_exp(acc_o_updater.curr_max_);
            }

            // Compute the sum.
            // float p_sum[Softmax::ROWS_PER_THREAD];
            softmax.template reduce<fmha::Sum_>(acc_o_updater.prev_sum_);

            // Trigger the load for the next K/V values.
            if (kv_loop + Cta_tile_p::N < kv_loop_bound && !PREFETCH_KV_BUFFER_TO_SMEM)
            {

                gmem_k.move();
                gmem_v.move();
                smem_k.move_to_next_write_buffer();
                smem_v.move_to_next_write_buffer();
                gmem_k.load(smem_k);
                gmem_v.load(smem_v);

                // Push the LDGDEPBAR instruction after the loads for Q, K and V.
                fmha::ldgdepbar<USE_LDGSTS>();

                gmem_k.commit(smem_k);
                if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
                {
                    gmem_v.commit(smem_v);
                }
            }

            // Update acc_sum scale of flash attention
            acc_o_updater.update_acc_sum();

            // Finalize softmax on the accumulators of P^T.
            // NOTE: don't scale here (scale finally)
            // softmax.scale(p_sum);

            // Store the P matrix.
#if defined(STORE_S)
            softmax.store(gmem_s);
#endif

#if defined(STORE_P)
            gmem_p.move();
#endif

#if defined(STORE_S)
            gmem_s.move();
#endif

            if (kv_loop + Cta_tile_p::N < kv_loop_bound)
            {

                fmha::depbar<USE_LDGSTS, 1>();
                __syncthreads();

                smem_k.move_to_next_read_buffer();
                if (Kernel_traits::LIMIT_QK_FRAGMENTS)
                {
                    smem_k.load(frag_k[0], 0);
                }
                else
                {
#pragma unroll
                    for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
                    {
                        smem_k.load(frag_k[ki], ki);
                    }
                }
            }

            // Repack for the next BMM.
            fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
            softmax.pack(frag_p);

            fmha::Fragment_accumulator<Traits_o> local_acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
            using Acc_type_o = typename Traits_o::Accumulator_type;
            fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(local_acc_o);

// Do this part of O = P^T * V^T.
#pragma unroll
            for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
            {
                fmha::gemm(local_acc_o, frag_p[ki], frag_v[ki]);
            }

            // Update acc_o of flash attention
            acc_o_updater.update_o(acc_o, local_acc_o);

            if (kv_loop + Cta_tile_p::N < kv_loop_bound)
            {

                // Commit the data for V to shared memory if it has not been done already.
                if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
                {
                    // Make sure we are done loading the fragments for K.
                    __syncthreads();

                    // Commit the data to shared memory for V.
                    gmem_v.commit(smem_v);

                    // Make sure the data is in shared memory.
                    __syncthreads();
                }

                smem_v.move_next_read_buffer();

#pragma unroll
                for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
                {
                    smem_v.load(frag_v[ki], ki);
                }
            }
        } // Inner loop over the key/value sequence length.

        // Trigger the load for the next Q, K, V values.
        if (q_loop + Cta_tile_p::M < q_loop_bound)
        {
            smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
            smem_k.move_to_next_write_buffer();
            smem_v.move_to_next_write_buffer();
            gmem_k.reset();
            gmem_v.reset();
            if (!USE_LDGSTS_KV)
            {
                gmem_k.load(smem_k);
                gmem_v.load(smem_v);
            }
        }

        // Make sure we have the LDGDEPBAR in place.
        fmha::ldgdepbar<USE_LDGSTS>();

// Loop over MMAS_M.
// NOTE: o and might shared the same buffer with kv
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
            __syncthreads();

            // Output the values.
            gmem_o.store(out, ii);
        }

        // Move to the next part of the output.
        gmem_o.move();

        // Trigger the load for the next Q, K, V values.
        if (q_loop + Cta_tile_p::M < q_loop_bound)
        {
            if (USE_LDGSTS_KV)
            {
                gmem_k.load(smem_k);
                gmem_v.load(smem_v);
            }

            // Make sure we have the LDGDEPBAR in place.
            fmha::ldgdepbar<USE_LDGSTS>();

            // Commit the data for Q, K, V to shared memory.
            gmem_q.commit(smem_q);

            gmem_k.commit(smem_k);

            if (!Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
            {
                gmem_v.commit(smem_v);
            }
        }

        // Make sure the data is in shared memory.
        fmha::depbar<USE_LDGSTS, 1>();
        __syncthreads();

        // Make sure we are reading from the correct buffer.
        if (USE_LDGSTS_Q)
        {
            smem_q.move_to_next_read_buffer();
        }

// Trigger the loads for the values of Q for the next iteration.
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
        {
            smem_q.load(frag_q[ki], ki);
        }

        if (USE_LDGSTS_K)
        {
            smem_k.move_to_next_read_buffer();
        }

        // Trigger the loads for the values of K, V for the next iteration.
        if (Kernel_traits::LIMIT_QK_FRAGMENTS)
        {
            smem_k.load(frag_k[0], 0);
        }
        else
        {
#pragma unroll
            for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki)
            {
                smem_k.load(frag_k[ki], ki);
            }
        }

        // Commit the data for V to shared memory if it has not been done already.
        if (Kernel_traits::SHARE_SMEM_FOR_K_AND_V)
        {
            // Make sure we are done loading the fragments for K.
            __syncthreads();

            // Commit the data to shared memory for V.
            gmem_v.commit(smem_v);

            // Make sure the data is in shared memory.
            __syncthreads();
        }

        if (USE_LDGSTS_V)
        {
            smem_v.move_to_next_read_buffer();
        }

#pragma unroll
        for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
        {
            smem_v.load(frag_v[ki], ki);
        }

    } // Outer loop over the query sequence length.

} // device_flash_attention_1xN

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
