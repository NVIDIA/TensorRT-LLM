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
#include <fmha/hopper/fragment.h>

namespace fmha
{
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Smem_tile_a, typename Smem_tile_b,
    bool GMMA_A_RF_, // GMMA A operand coming from RF?
    bool GMMA_B_RF_  // GMMA B operand coming from RF?
    >
struct Compute_tile_with_gmma
{
};

/*
compute tile used when both operands are coming from SMEM
*/
template <typename Traits, typename Cta_tile, typename Smem_tile_a, typename Smem_tile_b>
struct Compute_tile_with_gmma<Traits, Cta_tile, Smem_tile_a, Smem_tile_b,
    false, // GMMA A operand coming from SMEM
    false  // GMMA B operand coming from SMEM
    >
{

    static constexpr int NUM_KBLOCKS = Smem_tile_b::BUFFERS_PER_TILE / Cta_tile::WARPS_K;
    static_assert(NUM_KBLOCKS * Cta_tile::WARPS_K == Smem_tile_b::BUFFERS_PER_TILE);
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // desc for A and B should have the same strategy
    static_assert(Smem_tile_a::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP
            == Smem_tile_b::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP,
        "GMMA desc for A and B should have the same strategy.");

    // The number of MMAs.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    enum
    {
        MMAS_K = Mma_tile::MMAS_K
    };

    // Ctor.
    inline __device__ Compute_tile_with_gmma() {}

    // Ctor, that helps set the gmma descs to support different buffer index as the start address.
    inline __device__ Compute_tile_with_gmma(void* a_smem_, void* b_smem_)
        : Compute_tile_with_gmma(__nvvm_get_smem_pointer(a_smem_), __nvvm_get_smem_pointer(b_smem_))
    {
    }

    inline __device__ Compute_tile_with_gmma(uint32_t a_smem_base, uint32_t b_smem_base)
        : a_smem_base_(a_smem_base)
        , b_smem_base_(b_smem_base)
    {

        // We always start at buffer 0.
        uint32_t a_smem = a_smem_base_;
        uint32_t b_smem = b_smem_base_;

#pragma unroll
        for (int mma_m_idx = 0; mma_m_idx < MMAS_M; ++mma_m_idx)
        {
            gmma_desc_a_[mma_m_idx].set_smem_pointer(a_smem + mma_m_idx * Smem_tile_a::GMMA_GROUP_SMEM_DISTANCE);
            // We take the number of buffers directly from the Smem_tile. If we have only one buffer, the return offset
            // is 0.
            gmma_desc_a_[mma_m_idx].set_max_descriptor_0(
                Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB * (Smem_tile_a::BUFFERS_PER_TILE - 1));
        }

#pragma unroll
        for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
        {
            gmma_desc_b_[mma_n_idx].set_smem_pointer(b_smem + mma_n_idx * Smem_tile_b::GMMA_GROUP_SMEM_DISTANCE);
            gmma_desc_b_[mma_n_idx].set_max_descriptor_0(
                Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB * (Smem_tile_b::BUFFERS_PER_TILE - 1));
        }
    }

    // move the gmme desc by N buffers.
    //  Something nice to have if we have persistent kernels.
    inline __device__ void increment_N_gmma_desc_group(int N)
    {
#pragma unroll
        for (int idx = 0; idx < Smem_tile_a::Gmma_descriptor::NUM_DESCRIPTORS; ++idx)
        {
#pragma unroll
            for (int mma_m_idx = 0; mma_m_idx < MMAS_M; ++mma_m_idx)
            {
                uint64_t temp_desc = gmma_desc_a_[mma_m_idx].get_descriptor(idx);
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                tmp.x = (tmp.x & 0xFFFF0000) + (a_smem_base_ / 16)
                    + mma_m_idx * Smem_tile_a::GMMA_GROUP_SMEM_DISTANCE / 16
                    + N * Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                gmma_desc_a_[mma_m_idx].set_descriptor(idx, temp_desc);
            }

#pragma unroll
            for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
            {
                uint64_t temp_desc = gmma_desc_b_[mma_n_idx].get_descriptor(idx);
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                tmp.x = (tmp.x & 0xFFFF0000) + (b_smem_base_ / 16) + N * Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                gmma_desc_b_[mma_n_idx].set_descriptor(idx, temp_desc);
            }
        }
    }

    // Clear the accumulators. It does nothing as we have a special flag for GMMA.
    inline __device__ void clear()
    {
        fmha::clear(acc_);
    }

    // smarter way of increment a group of gmma desc.
    // if one of them need to be reset to the first ldgsts buffer
    // it is very likely (currently guaranteed) that all of them need to be reset to the first
    // ldgsts buffer.
    // we do this to save the usage of uniform register. Otherwise, kernel with larger M could not
    // achieve sol.
    inline __device__ void increment_gmma_desc_group()
    {
        bool reset_buffer_a = gmma_desc_a_[0].get_descriptor(0) >= gmma_desc_a_[0].get_max_descriptor_0();
        bool reset_buffer_b = gmma_desc_b_[0].get_descriptor(0) >= gmma_desc_b_[0].get_max_descriptor_0();

#pragma unroll
        for (int idx = 0; idx < Smem_tile_a::Gmma_descriptor::NUM_DESCRIPTORS; ++idx)
        {
#pragma unroll
            for (int mma_m_idx = 0; mma_m_idx < MMAS_M; ++mma_m_idx)
            {
                uint64_t temp_desc = gmma_desc_a_[mma_m_idx].get_descriptor(idx);
                // smem start address is in lower 32bits
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                if (reset_buffer_a)
                {
                    tmp.x -= (Smem_tile_a::BUFFERS_PER_TILE - 1) * Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                }
                else
                {
                    tmp.x += Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                }

                gmma_desc_a_[mma_m_idx].set_descriptor(idx, temp_desc);
            }

#pragma unroll
            for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
            {
                uint64_t temp_desc = gmma_desc_b_[mma_n_idx].get_descriptor(idx);
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                if (reset_buffer_b)
                {
                    tmp.x -= (Smem_tile_b::BUFFERS_PER_TILE - 1) * Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                }
                else
                {
                    tmp.x += Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                }
                gmma_desc_b_[mma_n_idx].set_descriptor(idx, temp_desc);
            }
        }
    }

    // smarter way of increment a group of gmma desc.
    // if one of them need to be reset to the first ldgsts buffer
    // it is very likely (currently guaranteed) that all of them need to be reset to the first
    // ldgsts buffer.
    // we do this to save the usage of uniform register. Otherwise, kernel with larger M could not
    // achieve sol.
    inline __device__ void increment_gmma_desc_a_group()
    {
        bool reset_buffer = gmma_desc_a_[0].get_descriptor(0) >= gmma_desc_a_[0].get_max_descriptor_0();

#pragma unroll
        for (int idx = 0; idx < Smem_tile_b::Gmma_descriptor::NUM_DESCRIPTORS; ++idx)
        {
#pragma unroll
            for (int mma_m_idx = 0; mma_m_idx < MMAS_M; ++mma_m_idx)
            {
                uint64_t temp_desc = gmma_desc_a_[mma_m_idx].get_descriptor(idx);
                // smem start address is in lower 32bits
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                if (reset_buffer)
                {
                    tmp.x -= (Smem_tile_a::BUFFERS_PER_TILE - 1) * Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                }
                else
                {
                    tmp.x += Smem_tile_a::BYTES_PER_BUFFER_NO_4LSB;
                }
                gmma_desc_a_[mma_m_idx].set_descriptor(idx, temp_desc);
            }
        }
    }

    // smarter way of increment a group of gmma desc.
    // if one of them need to be reset to the first ldgsts buffer
    // it is very likely (currently guaranteed) that all of them need to be reset to the first
    // ldgsts buffer.
    // we do this to save the usage of uniform register. Otherwise, kernel with larger M could not
    // achieve sol.
    template <bool RESET_CHECK = true>
    inline __device__ void increment_gmma_desc_b_group(int N = 1)
    {
        bool reset_buffer = RESET_CHECK && gmma_desc_b_[0].get_descriptor(0) >= gmma_desc_b_[0].get_max_descriptor_0();

#pragma unroll
        for (int idx = 0; idx < Smem_tile_b::Gmma_descriptor::NUM_DESCRIPTORS; ++idx)
        {
#pragma unroll
            for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
            {
                uint64_t temp_desc = gmma_desc_b_[mma_n_idx].get_descriptor(idx);
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                if (reset_buffer)
                {
                    tmp.x -= (Smem_tile_b::BUFFERS_PER_TILE - 1) * Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                }
                else
                {
                    tmp.x += Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                }
                gmma_desc_b_[mma_n_idx].set_descriptor(idx, temp_desc);
            }
        }
    }

    // Compute.
    // last of group indicates it is the last GMMA with a GMMA group. So the GSB should be updated
    // last of kblock indicates it is the last GMMA with kblock. so desc will be updated accordingly
    inline __device__ void compute(int ki, bool last_of_group = false, bool last_of_kblock = false)
    {
#pragma unroll
        for (int mmas_m_idx = 0; mmas_m_idx < MMAS_M; ++mmas_m_idx)
        {
#pragma unroll
            for (int mmas_n_idx = 0; mmas_n_idx < MMAS_N; ++mmas_n_idx)
            {
                // weird code to use SEL to avoid reg spill
                typename Smem_tile_a::Gmma_descriptor::Single_desc single_desc_a;
                typename Smem_tile_b::Gmma_descriptor::Single_desc single_desc_b;

                single_desc_a.set(gmma_desc_a_[mmas_m_idx].get_descriptor(ki));
                single_desc_b.set(gmma_desc_b_[mmas_n_idx].get_descriptor(ki));

                if (mmas_n_idx == (MMAS_N - 1))
                {
                    // update desc for A
                    gmma_desc_a_[mmas_m_idx].increment_single_descriptor(last_of_kblock);
                }
                if (mmas_m_idx == (MMAS_M - 1))
                {
                    // update desc for B
                    gmma_desc_b_[mmas_n_idx].increment_single_descriptor(last_of_kblock);
                }

                if ((last_of_group == true) && (mmas_m_idx == (MMAS_M - 1)) && (mmas_n_idx == (MMAS_N - 1)))
                {
                    // increment the scoreboard
                    acc_[mmas_m_idx][mmas_n_idx].template mma<true>(single_desc_a, single_desc_b);
                }
                else
                {
                    acc_[mmas_m_idx][mmas_n_idx].template mma<false>(single_desc_a, single_desc_b);
                }
            } // for (mmas_n_idx)
        }     // for (mmas_m_idx)
    }

    // Load from shared memory. For GMMA where both operand comes from SMEM, this does nothing
    inline __device__ void load(Smem_tile_a& smem_a, Smem_tile_b& smem_b, int ki, bool first = false) {}

    // The accumulators.
    Fragment_accumulator<Traits> acc_[MMAS_M][MMAS_N];

    // one descriptor group per stage, different GMMAs may or maynot share descriptor group
    // each descriptor group holds all the descriptors for the entire kblock

    // The descriptor to load A.
    typename Smem_tile_a::Gmma_descriptor gmma_desc_a_[MMAS_M];
    // The descriptor to load B.
    typename Smem_tile_b::Gmma_descriptor gmma_desc_b_[MMAS_N];
    uint32_t a_smem_base_, b_smem_base_;
};

/*
compute tile used when A is from RF, B is from SMEM
*/
template <typename Traits, typename Cta_tile, typename Smem_tile_a, typename Smem_tile_b>
struct Compute_tile_with_gmma<Traits, Cta_tile, Smem_tile_a, Smem_tile_b,
    true, // GMMA A operand coming from RF
    false // GMMA B operand coming from SMEM
    >
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The fragment for holding A.
    using Fragment = Fragment_a<Traits, Row>;

    // static_assert(Cta_tile::K == 128);
    // static_assert(Mma_tile::K_PER_MMA_PER_CTA == 64 );
    // pstatic_assert(NUM_KBLOCKS == 384 / 64);
    static constexpr int NUM_KBLOCKS = Smem_tile_b::BUFFERS_PER_TILE / Cta_tile::WARPS_K;
    // static_assert(NUM_KBLOCKS * Cta_tile::WARPS_K == Smem_tile_b::BUFFERS_PER_TILE);

    // desc for A and B should have the same strategy
    static_assert(Smem_tile_a::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP
            == Smem_tile_b::Gmma_descriptor::GMMA_DESC_SIZE_PER_GROUP,
        "GMMA desc for A and B should have the same strategy.");

    // The number of MMAs.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    enum
    {
        MMAS_N = Mma_tile::MMAS_N
    };

    // TODO
    enum
    {
        MMAS_K = Mma_tile::MMAS_K * Cta_tile::WARPS_K
    };

    // Ctor.
    inline __device__ Compute_tile_with_gmma() {}

    // Ctor, that helps set the gmma descs
    inline __device__ Compute_tile_with_gmma(void* a_smem_, void* b_smem_)
        : Compute_tile_with_gmma(__nvvm_get_smem_pointer(a_smem_), __nvvm_get_smem_pointer(b_smem_))
    {
    }

    inline __device__ Compute_tile_with_gmma(uint32_t, uint32_t b_smem_base)
        : b_smem_base_(b_smem_base)
    {

        // We always start at buffer 0 and take the number of buffers from the Smem_tile, as above.
        uint32_t b_smem = b_smem_base_;
// do not need to set desc for matrix A
#pragma unroll
        for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
        {
            gmma_desc_b_[mma_n_idx].set_smem_pointer(b_smem + mma_n_idx * Smem_tile_b::GMMA_GROUP_SMEM_DISTANCE);
            gmma_desc_b_[mma_n_idx].set_max_descriptor_0(
                Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB * (Smem_tile_b::BUFFERS_PER_TILE - 1));
        }
    }

    // move the gmme desc by N buffers.
    //  Something nice to have if we have persistent kernels.
    inline __device__ void increment_N_gmma_desc_group(int N)
    {
#pragma unroll
        for (int idx = 0; idx < Smem_tile_b::Gmma_descriptor::NUM_DESCRIPTORS; ++idx)
        {
#pragma unroll
            for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
            {
                uint64_t temp_desc = gmma_desc_b_[mma_n_idx].get_descriptor(idx);
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                tmp.x = (tmp.x & 0xFFFF0000) + (b_smem_base_ / 16) + (N) *Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                gmma_desc_b_[mma_n_idx].set_descriptor(idx, temp_desc);
            }
        }
    }

    // Clear the accumulators. It does nothing as we have a special flag for GMMA.
    inline __device__ void clear()
    {
        fmha::clear(acc_);
    }

    // smarter way of increment a group of gmma desc.
    // if one of them need to be reset to the first ldgsts buffer
    // it is very likely (currently guaranteed) that all of them need to be reset to the first
    // ldgsts buffer.
    // we do this to save the usage of uniform register. Otherwise, kernel with larger M could not
    // achieve sol.

    template <bool RESET_CHECK = true>
    inline __device__ void increment_gmma_desc_group(int N = 1)
    {
        bool reset_buffer = RESET_CHECK && gmma_desc_b_[0].get_descriptor(0) >= gmma_desc_b_[0].get_max_descriptor_0();

#pragma unroll
        for (int idx = 0; idx < Smem_tile_b::Gmma_descriptor::NUM_DESCRIPTORS; ++idx)
        {
#pragma unroll
            for (int mma_n_idx = 0; mma_n_idx < MMAS_N; ++mma_n_idx)
            {
                uint64_t temp_desc = gmma_desc_b_[mma_n_idx].get_descriptor(idx);
                int2& tmp = reinterpret_cast<int2&>(temp_desc);
                if (reset_buffer)
                {
                    tmp.x -= (Smem_tile_b::BUFFERS_PER_TILE - 1) * Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                }
                else
                {
                    tmp.x += Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB;
                }
                gmma_desc_b_[mma_n_idx].set_descriptor(idx, temp_desc);
            }
        }
    }

    // Compute.
    // last of group indicates it is the last GMMA with a GMMA group. So the GSB should be updated
    // last of kblock indicates it is the last GMMA with kblock. so desc will be updated accordingly
    inline __device__ void compute(int ki, bool last_of_group = false, bool last_of_kblock = false)
    {

#pragma unroll
        for (int mmas_m_idx = 0; mmas_m_idx < MMAS_M; ++mmas_m_idx)
        {
#pragma unroll
            for (int mmas_n_idx = 0; mmas_n_idx < MMAS_N; ++mmas_n_idx)
            {
                // weird code to use SEL to avoid reg spill
                typename Smem_tile_b::Gmma_descriptor::Single_desc single_desc_b;

                single_desc_b.set(gmma_desc_b_[mmas_n_idx].get_descriptor(ki));

                if (mmas_m_idx == (MMAS_M - 1))
                {
                    // update desc for B
                    gmma_desc_b_[mmas_n_idx].increment_single_descriptor(last_of_kblock);
                }

                if ((last_of_group == true) && (mmas_m_idx == (MMAS_M - 1)) && (mmas_n_idx == (MMAS_N - 1)))
                {
                    // increment the scoreboard
                    acc_[mmas_m_idx][mmas_n_idx].template mma<true>(a_[mmas_m_idx], single_desc_b);
                }
                else
                {
                    acc_[mmas_m_idx][mmas_n_idx].template mma<false>(a_[mmas_m_idx], single_desc_b);
                }
            } // for (mmas_n_idx)
        }     // for (mmas_m_idx)
    }

    template <int K>
    inline __device__ void compute_incta_splitk(Fragment const (&frag_a)[K][1], int const warp_k)
    {

        if (Smem_tile_b::Gmma_descriptor::TRANS_MODE == Gmma_descriptor_transpose::NOTRANS)
        {
            // In this case, the K dimension is the leading dimension, so we need to set the smem locations correctly
            // for each Warp in K.

            // The number of elements in K per group.
            constexpr int ELTS_PER_KGROUP = Smem_tile_b::BYTES_PER_ROW / sizeof(typename Traits::B_type);
            // The number of MMAS to perform before incrementing by the group stride.
            constexpr int MMAS_K_PER_GROUP = ELTS_PER_KGROUP / Traits::GMMA_K;
            // The number of MMAS a k-warp performs.
            constexpr int MMAS_K_PER_WARP = Mma_tile::MMAS_K;

            int const group_offset = warp_k * MMAS_K_PER_WARP;
            // Initialize the descriptor
            int gi = group_offset / MMAS_K_PER_GROUP;
            int ii = group_offset % MMAS_K_PER_GROUP;

            int BYTES_OFFSET_NO_4LSB = gi * Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB
                + ii * Smem_tile_b::Gmma_descriptor::BYTES_PER_DESC_NO_4LSB;

            uint64_t desc_b = gmma_desc_b_[0].get_descriptor(0);
            int2& desc_b_view = reinterpret_cast<int2&>(desc_b);
            desc_b_view.x += BYTES_OFFSET_NO_4LSB;

            typename Smem_tile_b::Gmma_descriptor::Single_desc single_desc_b;
            single_desc_b.set(desc_b);
#pragma unroll
            for (int ki = 0; ki < MMAS_K_PER_WARP - 1; ki++)
            {
                acc_[0][0].template mma<false>(frag_a[ki][0], single_desc_b);

                // Increment the descriptor for the next kblock.
                int const ki_next = group_offset + ki + 1;
                // Update descriptor for next GMMA.
                if (ki_next % MMAS_K_PER_GROUP == 0)
                {
                    desc_b_view.x += Smem_tile_b::BYTES_PER_BUFFER_NO_4LSB
                        - Smem_tile_b::Gmma_descriptor::BYTES_DESC_INC_BOUNDARY_NO_4LSB;
                }
                else
                {
                    desc_b_view.x += Smem_tile_b::Gmma_descriptor::BYTES_PER_DESC_NO_4LSB;
                }
                single_desc_b.set(desc_b);
            }
            // Last one increments gsb.
            acc_[0][0].template mma<true>(frag_a[MMAS_K_PER_WARP - 1][0], single_desc_b);
        }
        else
        { // GMMA supports transposed input: we can just advance SMEM address to the k-th block for each Warp in K.

            constexpr int NUM_KGROUPS = Smem_tile_b::BUFFERS_PER_TILE;
            constexpr int MMAS_K_PER_GROUP = Mma_tile::MMAS_K / NUM_KGROUPS;
            static_assert(MMAS_K_PER_GROUP * NUM_KGROUPS == Mma_tile::MMAS_K);

            uint64_t temp_desc = gmma_desc_b_[0].get_descriptor(0);
            int2& tmp = reinterpret_cast<int2&>(temp_desc);

            constexpr int BYTES_PER_K_GROUP_NO_4LSB
                = Mma_tile::K_PER_WARP_GROUP * Mma_tile::N_PER_WARP_GROUP * sizeof(Traits::B_type) / 16;
            tmp.x += warp_k * BYTES_PER_K_GROUP_NO_4LSB;
            gmma_desc_b_[0].set_descriptor(0, temp_desc);

#pragma unroll
            for (int kbi = 0; kbi < NUM_KGROUPS - 1; kbi++)
            {
#pragma unroll
                for (int ki = 0; ki < MMAS_K_PER_GROUP; ki++)
                {
                    fill_frag_a(frag_a[kbi * MMAS_K_PER_GROUP + ki][0]);
                    // Never increment scoreboard, but check for last kblock.
                    compute(ki, false, ki == MMAS_K_PER_GROUP - 1);
                }
                increment_gmma_desc_group();
            }

#pragma unroll
            for (int ki = 0; ki < MMAS_K_PER_GROUP - 1; ki++)
            {
                fill_frag_a(frag_a[(NUM_KGROUPS - 1) * MMAS_K_PER_GROUP + ki][0]);
                compute(ki);
            }

            fill_frag_a(frag_a[NUM_KGROUPS * MMAS_K_PER_GROUP - 1][0]);
            compute(NUM_KGROUPS * MMAS_K_PER_GROUP - 1, true, true);
        }
    }

    // Fill the input fragment
    inline __device__ void fill_frag_a(Fragment a_temp)
    {
#pragma unroll
        for (int idx = 0; idx < Fragment::NUM_REGS; ++idx)
        {
            a_[0].reg(idx) = a_temp.reg(idx);
        }
    }

    // Load from shared memory.
    // we don't actually need this with MHA fused kernel.
    inline __device__ void load(Smem_tile_a& smem_a, Smem_tile_b& smem_b, int ki)
    {
        // smem_a.load( a_[ki], ki );
    }

    // The accumulators.
    Fragment_accumulator<Traits> acc_[MMAS_M][MMAS_N];

    // The fragments to load A.
    // Need to think about is is better to declare as Fragment a_?
    // for the second GEMM, MMAS_M is most likely 1. (at least for now. )
    Fragment a_[MMAS_M];

    // one descriptor group per stage, different GMMAs may or maynot share descriptor group
    // each descriptor group holds all the descriptors for the entire kblock

    // The descriptor to load B.
    typename Smem_tile_b::Gmma_descriptor gmma_desc_b_[MMAS_N];
    uint32_t b_smem_base_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
