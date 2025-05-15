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

inline __device__ void spin_wait_(int* barrier, int step, int expected)
{

    // THE FOLLOWING CODE MUST BE EXECUTED BY A SINGLE THREAD IN THE CTA.

    // Update the global counter. Make sure prior writes are visible.
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(barrier), "r"(step));

    // Busy wait. We could use found = old + step with old = atomicAdd(...) but it's not faster.
    for (int found = -1; found != expected;)
    {
        asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(barrier));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Functor, typename Params>
inline __device__ float reduce(float x, Params const& params, int* barrier, int bidx, int bidn, int tidx, bool inc)
{

    // The M dimension of the CTA tile.
    enum
    {
        M = Kernel_traits::Cta_tile_p::M
    };

    // Make sure it does not exceed the CTA size.
    static_assert(M <= Kernel_traits::Cta_tile_p::THREADS_PER_CTA, "");

    // The offset to beginning of the scratch space for that head.
    int const offset = bidx * Kernel_traits::CTAS_PER_HEAD * M + tidx;

    // The scratch buffer for that thread.
    float* scratch_ptr = Functor::IS_SUM ? params.sum_scratch_ptr : params.max_scratch_ptr;

    // Move to the beginning of the buffer.
    scratch_ptr += offset;

    // Active threads store their elements to global memory.
    if (tidx < M)
    {
        scratch_ptr[bidn * M] = x;
    }

    // The step to increment the counters.
    int step = inc ? 1 : -1;
    // The expected value.
    int expected = inc ? Kernel_traits::CTAS_PER_HEAD : 0;

    // Make sure the data is in memory.
    if (tidx == 0)
    {
        spin_wait_(barrier, step, expected);
    }

    // Make sure all the threads for the block leader (tidx == 0).
    __syncthreads();

    // Load the element from memory.
    if (bidn > 0 && tidx < M)
    {
        x = *scratch_ptr;
    }

    // Each CTA does the parallel reduction of values.
    for (int ii = 1; ii < Kernel_traits::CTAS_PER_HEAD; ++ii)
    {
        if (tidx < M)
        {
            x = Functor::apply(x, scratch_ptr[ii * M]);
        }
    }

    // The final value.
    return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int CTAS_PER_HEAD>
inline __device__ void acquire_lock_(int* lock, int bidn, int tidx)
{

    // THE FOLLOWING CODE MUST BE EXECUTED BY A SINGLE THREAD IN THE CTA.

    // Poll.
    for (int found = -1; found != bidn;)
    {
        asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(lock));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int CTAS_PER_HEAD>
inline __device__ void release_lock_(int* lock, int bidn, int tidx)
{

    // THE FOLLOWING CODE MUST BE EXECUTED BY A SINGLE THREAD IN THE CTA.

    // Update the global counter. The last CTA resets the counter.
    if (bidn == CTAS_PER_HEAD - 1)
    {
        asm volatile("st.global.release.gpu.b32 [%0], 0;" ::"l"(lock));
    }
    else
    {
        // atomicAdd(lock, 1);
        asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(lock));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void device_1xN_multi_cta(Params const& params)
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

    // Do we use LDGSTS for any of the 3 input matrices.
    enum
    {
        USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V
    };

    // If either K or V uses LDGSTS, they cannot share a buffer.
    static_assert(!(USE_LDGSTS_K || USE_LDGSTS_V) || !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");

    // Shared memory.
    extern __shared__ char smem_[];

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

    // The block index for the batch * head.
    int const bidx = blockIdx.y;
    // The block index for the multi-CTA distribution.
    int const bidn = blockIdx.x;
    // The thread index.
    int const tidx = threadIdx.x;

    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Do we increase/decrease the barrier count.
    bool barrier_inc = true;

    // Outer persistent loop.
    for (int bidbh = bidx; bidbh < params.b * params.h; bidbh += params.heads_per_wave)
    {

        // Decompose the index into b/h. TODO: Should we use fast divmod?
        int bidb = bidbh / params.h;
        int bidh = bidbh % params.h;

        // The special structure to control the block info.
        Multi_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, bidn, tidx);
        if (bidb >= params.b || binfo.stop_early())
        {
            return;
        }

        // Create the object to control the masks.
        fmha::Mask<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION> mask(params, binfo, tidx);

        // Allocate the global memory tile loader for Q.
        Gmem_tile_q gmem_q(params, 0, binfo, tidx);
        // Allocate the global memory tile loader for K.
        Gmem_tile_k gmem_k(params, 1, binfo, tidx, bidn * Cta_tile_p::N);
        // Allocate the global memory tile loader for V.
        Gmem_tile_v gmem_v(params, 2, binfo, tidx, bidn * Cta_tile_p::N);
        // Allocate the global memory writer for O. Does not depend on bidn as we do a reduction.
        Gmem_tile_o gmem_o(params, binfo, tidx);

        // Trigger the loads for Q.
        gmem_q.load(smem_q);
        // Trigger the loads for K.
        gmem_k.load(smem_k);
        // Trigger the loads for V.
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
        fmha::depbar_<USE_LDGSTS>();
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

        // Create the object to do the softmax.
        using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
        Softmax softmax(params, &smem_[Smem_tile_q::BYTES_PER_TILE], bidb, tidx);

        // Load over the entire sequence length.
        for (int loop = 0, outer = 0; loop < binfo.actual_seqlen; loop += Cta_tile_p::M, outer++)
        {

            // If we have reached the length of the sequence, stop earlier.
            if (loop >= binfo.actual_seqlen)
            {
                break;
            }

            // Declare the accumulators for the 1st gemm.
            fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
            using Acc_type_p = typename Traits_p::Accumulator_type;
            fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
            for (int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki)
            {

                // Trigger the load from shared memory for the next series of Q values.
                smem_q.load(frag_q[ki & 1], ki);
                // Do the math for the values already in registers.
                if (ki <= Mma_tile_p::VALID_MMAS_K)
                {
                    fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
                }
            }

            // Do the final stage of math.
            if (Mma_tile_p::MMAS_K <= Mma_tile_p::VALID_MMAS_K)
            {
                int ki = Mma_tile_p::MMAS_K;
                fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
            }

            // Load the mask for that iteration.
            mask.load(outer);

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

            // Make sure we are done reading the data (for K).
            __syncthreads();

            // Enable our trick to use the max for INT8 to scale. 16129 == 127 ^ 2.
            if (Kernel_traits::USE_SCALE_MAX)
            {
                float p_max = reinterpret_cast<float const&>(params.scale_bmm1) * 16129.f;
                softmax.apply_exp(p_max);
            }
            else
            {
                // Compute the max.
                float partial = softmax.template reduce_<fmha::Max_>();
                float total = reduce<Kernel_traits, fmha::Max_>(
                    partial, params, &params.max_barriers[bidx], bidx, bidn, tidx, barrier_inc);

                // The global reduction calls syncthreads so we know that shared memory was read.
                // __syncthreads();

                // Reshuffle the max amongst threads.
                float p_max[Softmax::ROWS_PER_THREAD];
                softmax.shuffle(p_max, total);

                // Compute the exponential value.
                softmax.apply_exp(p_max);
            }

            // The softmax.shuffle function ends with __syncthreads(). No need for an extra sync!
            // __syncthreads();

            // Compute the sum.
            float partial = softmax.template reduce_<fmha::Sum_>();
            float total = reduce<Kernel_traits, fmha::Sum_>(
                partial, params, &params.sum_barriers[bidx], bidx, bidn, tidx, barrier_inc);
            // Switch.
            barrier_inc = !barrier_inc;

            // The global reduction calls syncthreads so we know that shared memory was read.
            // __syncthreads();

            // Reshuffle the sum amongst threads.
            float p_sum[Softmax::ROWS_PER_THREAD];
            softmax.shuffle(p_sum, total);

            // The softmax.shuffle function ends with __syncthreads(). No need for an extra sync!
            // __syncthreads();

            // Finalize softmax on the accumulators of P^T.
            softmax.scale(p_sum);

            // Trigger the load for the next Q values.
            int do_prefetching = loop + Cta_tile_p::M < binfo.actual_seqlen;
            if (do_prefetching)
            {
                smem_q.move_to_next_write_buffer();
                gmem_q.move();
                gmem_q.load(smem_q);
            }

            // Make sure we have the LDGDEPBAR in place.
            fmha::ldgdepbar<USE_LDGSTS_Q>();

            // Repack for the next BMM.
            fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
            softmax.pack(frag_p);

            // Declare the accumulators for the 1st gemm.
            fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
            using Acc_type_o = typename Traits_o::Accumulator_type;
            fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

// Do this part of O = P^T * V^T.
#pragma unroll
            for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki)
            {
                fmha::gemm(acc_o, frag_p[ki], frag_v[ki]);
            }

            // Declare a lock.
            int* lock = &params.locks[bidx];

// Loop over MMAS_M.
#pragma unroll
            for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii)
            {

                // Enter the critical section.
                if (tidx == 0)
                {
                    acquire_lock_<Kernel_traits::CTAS_PER_HEAD>(lock, bidn, tidx);
                }

                // Make sure the lock was acquired by the leader. And protect SMEM from previous iter.
                __syncthreads();

                // Load the previous values from memory.
                uint4 old[Gmem_tile_o::STGS_PER_LOOP];
                gmem_o.load(old, ii);

                // Swizzle the elements and do the final reduction.
                smem_o.store(acc_o, ii);

                // Make sure the data is in shared memory.
                __syncthreads();

                // Load from shared memory.
                uint4 out[Gmem_tile_o::STGS_PER_LOOP];
                smem_o.load(out);

                // Output the values.
                gmem_o.store(out, old, ii);

                // Make sure the threads are done.
                __syncthreads();

                // Leave the critical section.
                if (tidx == 0)
                {
                    release_lock_<Kernel_traits::CTAS_PER_HEAD>(lock, bidn, tidx);
                }
            }

            // Move to the next part of the output.
            gmem_o.move();

            // Commit the values for Q into shared memory.
            if (do_prefetching)
            {
                gmem_q.commit(smem_q);
            }

            // Make sure we are reading from the correct buffer.
            if (USE_LDGSTS_Q && do_prefetching)
            {
                smem_q.move_to_next_read_buffer();
            }

            // Make sure the data is in shared memory.
            fmha::depbar_<USE_LDGSTS_Q>();
            __syncthreads();

            // Trigger the loads for the values of Q for the next iteration.
            if (do_prefetching)
            {
                smem_q.load(frag_q[0], 0);
            }

        } // Loop over the sequence length

        // Make sure we can start reusing shared memory.
        __syncthreads();

    } // Loop over the heads
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fused_multihead_attention
