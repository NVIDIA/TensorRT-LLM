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

namespace fmha
{
////////////////////////////////////////////////////////////////////////////////////////////////////
// whether transpose is applied on the smem before GMMA math execution
// if TN, notrans is applied to both A and B. as GMMA expects the data
// to be in TN format.
// if NT, trans is applied to both A and B.
////////////////////////////////////////////////////////////////////////////////////////////////////
enum class Gmma_descriptor_transpose
{
    TRANS,
    NOTRANS
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gmma descriptor mode
// 2 bits to specify the descriptor mode.
////////////////////////////////////////////////////////////////////////////////////////////////////
enum class Gmma_descriptor_mode
{
    SWIZZLE_NONE = 0,
    SWIZZLE_128B,
    SWIZZLE_64B,
    SWIZZLE_32B
};
constexpr uint32_t GMMA_DESCRIPTOR_MODE_BITS = 2;
constexpr uint32_t GMMA_DESCRIPTOR_MODE_SHIFT = 62;

////////////////////////////////////////////////////////////////////////////////////////////////////
// number of descriptor per GMMA group to be actually allocated per kblock
////////////////////////////////////////////////////////////////////////////////////////////////////
enum class Gmma_descriptor_size
{
    ONE,
    TWO, // not yet implemented. might be needed for 64xNxK tile size.
    // as many as needed (kblock / gmma_k). we may not prefer to use this as we may run out of UR
    // budget
    ALL
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// a single desc that has the info and bits
////////////////////////////////////////////////////////////////////////////////////////////////////
template <Gmma_descriptor_transpose Gmma_trans, Gmma_descriptor_mode Gmma_mode>
class Single_descriptor
{
public:
    // trans mode
    static constexpr Gmma_descriptor_transpose TRANS_MODE = Gmma_trans;

    // set the single desc
    inline __device__ void set(uint64_t const& desc_)
    {
        desc = desc_;
    }

    // get the single desc
    inline __device__ uint64_t get() const
    {
        return desc;
    }

private:
    // the descriptor, each of 64 bit
    uint64_t desc;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// for a
////////////////////////////////////////////////////////////////////////////////////////////////////

template <Gmma_descriptor_transpose Gmma_trans, Gmma_descriptor_mode Gmma_mode, typename Cta_tile, int BITS_PER_ELEMENT,
    int GMMA_M, int GMMA_N, int GMMA_K,
    // number of desc actually allocated.
    Gmma_descriptor_size Gmma_vector_size>
class Gmma_descriptor_a
{
public:
    // The type of the Single Descriptor
    using Single_desc = Single_descriptor<Gmma_trans, Gmma_mode>;

    // Transpose Mode
    static constexpr Gmma_descriptor_transpose TRANS_MODE = Gmma_trans;

    // The number of descriptors per 64xNblockxKblock.
    static constexpr Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = Gmma_vector_size;

    // Currently the number of descriptors per 64xNblockxKblock is always One
    // Historically we have supported more descriptors. But that has proven to
    // be less performant as it consumes too many uniform registers.
    // During the process of refactoring we have decided to only support allocating
    // one desc per 64xNblockxKblock. If needed in the future, we can support
    // more desc.
    static_assert(
        Gmma_vector_size == Gmma_descriptor_size::ONE, "Currently, only Mblock/64 desc is allocated per kgroup\n");

    // Interleaved Mode is currently not supported.
    // static_assert to avoid accidentally instantiate it.
    static_assert(
        Gmma_mode != Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, SWIZZLE_NONE mode is not implemented. \n");

    // byte per leading dim (row if TN, column is NT) must be 128
    enum
    {
        BYTES_PER_LEADING_DIM = 128
    };

    // bytes per element
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // the number of descriptors per kblock is related to GMMA shape and kblock size
    enum
    {
        NUM_DESCRIPTORS = (Gmma_vector_size == Gmma_descriptor_size::ALL) ? Cta_tile::K / GMMA_K : 1
    };

    // the number of descriptors per 128 byte in k dimension (leading dim)
    // NUM_DESCRIPTORS_PER_128B_IN_K is really only needed if leading dim is K
    enum
    {
        NUM_DESCRIPTORS_PER_128B_IN_K
        = (Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B && Gmma_trans == Gmma_descriptor_transpose::NOTRANS)
            ? BYTES_PER_LEADING_DIM / ((GMMA_K * BITS_PER_ELEMENT) / 8)
            : NUM_DESCRIPTORS
    };

    static constexpr uint32_t BYTES_PER_GMMA_K = GMMA_K * BITS_PER_ELEMENT / 8; // 32B

    // the distance between neighboring descriptors
    static constexpr uint32_t BYTES_PER_DESC = Gmma_vector_size == Gmma_descriptor_size::ALL ? 0
        : Gmma_trans == Gmma_descriptor_transpose::TRANS ? Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B
            ? GMMA_K * BYTES_PER_LEADING_DIM
            : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B ? (GMMA_K / 2) * BYTES_PER_LEADING_DIM
            : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_32B ? (GMMA_K / 4) * BYTES_PER_LEADING_DIM
                                                             : 0
        : Gmma_trans == Gmma_descriptor_transpose::NOTRANS
        ? Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B || Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B
            ? BYTES_PER_GMMA_K // 32B
            : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_32B ? Cta_tile::M * BYTES_PER_GMMA_K
                                                             : 0
        : 0;

    // the distance between neighboring desc without 4LSB
    static constexpr uint32_t BYTES_PER_DESC_NO_4LSB = BYTES_PER_DESC >> 4;

    // the distance to travel back from the last desc to the first desc within a group
    enum
    {
        BYTES_DESC_INC_BOUNDARY_NO_4LSB = BYTES_PER_DESC_NO_4LSB * (Cta_tile::K / GMMA_K - 1)
    };

    // set GMMA descriptor mode bits.
    static constexpr uint64_t DESCRIPTOR_MODE_IN_BIT_LOCATION
        = (static_cast<uint64_t>(Gmma_mode) & ((1u << GMMA_DESCRIPTOR_MODE_BITS) - 1)) << GMMA_DESCRIPTOR_MODE_SHIFT;

    // stride byte offset, bit 32-45, 4LSB not included
    // each row is always of 128 byte. 8 rows always.
    // divide by 16 since the 4 LSB is not included
    static constexpr uint64_t STRIDE_BYTE_OFFSET = BYTES_PER_LEADING_DIM
        * ((Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B)     ? 8
                : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                 : 2)
        / 16;
    // shift 32 bit
    static constexpr uint64_t STRIDE_BYTE_OFFSET_IN_BIT_LOCATION = STRIDE_BYTE_OFFSET << 32;

    // leading byte offset, bit 16-29, 4LSB not included
    // each row is still 128 byte.
    // divide by 16 since the 4 LSB is not included
    // for A matrix of TN, and the way we reshape the matrix, LEADING_BYTE_OFFSET is never non-zero
    // in the future with different GMMA shape, this might be needed
    static constexpr bool LEADING_BYTE_OFFSET_NEEDED = false;

    // the leading byte offset if needed 4LSB not included
    static constexpr uint64_t LEADING_BYTE_OFFSET = Gmma_mode == Gmma_descriptor_mode::SWIZZLE_32B
        ? BYTES_PER_LEADING_DIM / 16
        : BYTES_PER_LEADING_DIM * ((Gmma_trans == Gmma_descriptor_transpose::TRANS) ? Cta_tile::K : Cta_tile::M) / 16;
    // shift 16 bit
    static constexpr uint64_t LEADING_BYTE_OFFSET_IN_BIT_LOCATION
        = LEADING_BYTE_OFFSET_NEEDED ? LEADING_BYTE_OFFSET << 16 : 0;

    // ctor
    inline __device__ Gmma_descriptor_a()
    {
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] = 0;
        }

// set bit 62-63 to 1 for SWIZZLE_128B format
// set bit 62-63 to 2 for SWIZZLE_64B format
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] |= DESCRIPTOR_MODE_IN_BIT_LOCATION;
        }

// stride byte offset, bit 32-45, 4LSB not included
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] |= STRIDE_BYTE_OFFSET_IN_BIT_LOCATION;
        }

        // leading byte offset, bit 16-29, 4LSB not included
        if (LEADING_BYTE_OFFSET_NEEDED)
        {
#pragma unroll
            for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
            {
                desc[desc_idx] |= LEADING_BYTE_OFFSET_IN_BIT_LOCATION;
            }
        }
    }

    // update the descriptor based on smem address. Should be called once from prologue.
    inline __device__ void set_smem_pointer(uint32_t smem_nvvm_pointer)
    {
        // uint32_t smem_nvvm_pointer = get_smem_pointer(smem);
        uint64_t smem_address_bit = static_cast<uint64_t>(smem_nvvm_pointer);

        // set base offset, bit 49-61
        uint64_t offset = (smem_address_bit / BYTES_PER_LEADING_DIM)
            % ((Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B)     ? 8
                    : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                     : 2);
        uint64_t offset_in_bit_location = offset << 49;
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] |= offset_in_bit_location;
        }

// start_address, bit 0-13, 4LSB not included (so grab bit 4-17)
// the only bits that is different for each desc of the same obj
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            // for fp16, desc_idx_in_128B should range from 0 to 3
            int desc_idx_in_128B = desc_idx % NUM_DESCRIPTORS_PER_128B_IN_K;
            int desc_idx_over_128B = desc_idx / NUM_DESCRIPTORS_PER_128B_IN_K;

            uint64_t smem_address_bit_in_bit_location
                = (smem_address_bit + ((GMMA_K * BITS_PER_ELEMENT) / 8) * desc_idx_in_128B
                      + Cta_tile::M * BYTES_PER_LEADING_DIM * desc_idx_over_128B)
                << 46;

            smem_address_bit_in_bit_location = smem_address_bit_in_bit_location >> 50;
            desc[desc_idx] |= smem_address_bit_in_bit_location;
        }
    }

    // get a single desc from the desc group.
    inline __device__ uint64_t get_descriptor(int desc_idx) const
    {
        // printf("desc[0] = 0x%lx\n", desc[0]);
        return desc[(Gmma_vector_size == Gmma_descriptor_size::ALL) ? desc_idx : 0];
    }

    // get the max descriptor for desc[0]
    inline __device__ uint64_t get_max_descriptor_0() const
    {
        return max_desc_0;
    }

    // set a single desc from the desc group.
    inline __device__ void set_descriptor(int desc_idx, uint64_t single_desc)
    {
        desc[(Gmma_vector_size == Gmma_descriptor_size::ALL) ? desc_idx : 0] = single_desc;
    }

    // set the max descriptor for desc[0]. Should be called once from prologue.
    // Should be called with set_smem_pointer()
    // This value is needed to "loop back" to the first LDGSTS buffer when appropriate.
    inline __device__ void set_max_descriptor_0(int mem_offset_no_4LSB)
    {
        max_desc_0 = desc[0] + mem_offset_no_4LSB;
    }

    // for desc group where all desc all allocated,
    // increment_single_descriptor() will do nothing.
    inline __device__ void increment_single_descriptor(bool last_of_kblock)
    {
        // update smem start address, which is in lower 32bits.
        int2& tmp = reinterpret_cast<int2&>(desc[0]);
        if (last_of_kblock == true)
        {
            tmp.x -= BYTES_DESC_INC_BOUNDARY_NO_4LSB;
        }
        else
        {
            tmp.x += BYTES_PER_DESC_NO_4LSB;
        }
    }

    template <int BYTE_OFFSET>
    inline __device__ void increment_single_descriptor()
    {
        int2& tmp = reinterpret_cast<int2&>(desc[0]);
        tmp.x += (BYTE_OFFSET >> 4);
    }

private:
    // the descriptors, each of 64 bit
    uint64_t desc[NUM_DESCRIPTORS];
    // the max desc for desc_idx = 0
    uint64_t max_desc_0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// for b
////////////////////////////////////////////////////////////////////////////////////////////////////

template <Gmma_descriptor_transpose Gmma_trans, Gmma_descriptor_mode Gmma_mode, typename Cta_tile, int BITS_PER_ELEMENT,
    int GMMA_M, int GMMA_N, int GMMA_K,
    // number of desc actually allocated.
    Gmma_descriptor_size Gmma_vector_size>
class Gmma_descriptor_b
{
public:
    // The type of the Single Descriptor
    using Single_desc = Single_descriptor<Gmma_trans, Gmma_mode>;

    // Transpose mode.
    static constexpr Gmma_descriptor_transpose TRANS_MODE = Gmma_trans;

    // The number of descriptors per 64xNblockxKblock.
    static constexpr Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = Gmma_vector_size;

    // Currently the number of descriptors per 64xNblockxKblock is always One
    // Historically we have supported more descriptors. But that has proven to
    // be less performant as it consumes too many uniform registers.
    // During the process of refactoring we have decided to only support allocating
    // one desc per 64xNblockxKblock. If needed in the future, we can support
    // more desc.
    static_assert(
        Gmma_vector_size == Gmma_descriptor_size::ONE, "Currently, only Mblock/64 desc is allocated per kgroup\n");

    // Interleaved Mode is currently not supported.
    // static_assert to avoid accidentally instantiate it.
    static_assert(
        Gmma_mode != Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, SWIZZLE_NONE mode is not implemented. \n");

    // byte per leading dim (column if TN, row if NT), must be 128
    enum
    {
        BYTES_PER_LEADING_DIM = 128
    };

    // bytes per element
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // the number of descriptors per kblock is related to GMMA shape and kblock size
    enum
    {
        NUM_DESCRIPTORS = (Gmma_vector_size == Gmma_descriptor_size::ALL) ? Cta_tile::K / GMMA_K : 1
    };

    // the number of descriptors per 128 byte in k dimension (leading dim)
    // NUM_DESCRIPTORS_PER_128B_IN_K is really only needed if leading dim is K
    enum
    {
        NUM_DESCRIPTORS_PER_128B_IN_K
        = (Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B && Gmma_trans == Gmma_descriptor_transpose::NOTRANS)
            ? BYTES_PER_LEADING_DIM / ((GMMA_K * BITS_PER_ELEMENT) / 8)
            : NUM_DESCRIPTORS
    };

    static constexpr uint32_t BYTES_PER_GMMA_K = GMMA_K * BITS_PER_ELEMENT / 8; // 32B

    // the distance between neighboring descriptors
    static constexpr uint32_t BYTES_PER_DESC = Gmma_vector_size == Gmma_descriptor_size::ALL ? 0
        : Gmma_trans == Gmma_descriptor_transpose::TRANS ? Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B
            ? GMMA_K * BYTES_PER_LEADING_DIM
            : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B ? (GMMA_K / 2) * BYTES_PER_LEADING_DIM
            : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_32B ? (GMMA_K / 4) * BYTES_PER_LEADING_DIM
                                                             : 0
        : Gmma_trans == Gmma_descriptor_transpose::NOTRANS
        ? Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B || Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B
            ? BYTES_PER_GMMA_K // 32B
            : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_32B ? GMMA_N * BYTES_PER_GMMA_K
                                                             : 0
        : 0;

    // the distance between neighboring desc without 4LSB
    static constexpr uint32_t BYTES_PER_DESC_NO_4LSB = BYTES_PER_DESC >> 4;

    // the distance to travel back from the last desc to the first desc within a group
    enum
    {
        BYTES_DESC_INC_BOUNDARY_NO_4LSB = BYTES_PER_DESC_NO_4LSB * (Cta_tile::K / GMMA_K - 1)
    };

    // Byte count on tile-K dimension
    enum
    {
        RESET_SMEM = ((Gmma_trans == Gmma_descriptor_transpose::NOTRANS)
                         && (((Cta_tile::K * BITS_PER_ELEMENT) / (8 * BYTES_PER_LEADING_DIM)) > 1))
            ? true
            : false
    };

    // Reset bytes per BYTES_PER_LEADING_DIM (128) x tile-N
    enum
    {
        RESET_BYTES_NO_4LSB = (BYTES_PER_LEADING_DIM * Cta_tile::N) / 16
    };

    // set GMMA descriptor mode bits.
    static constexpr uint64_t DESCRIPTOR_MODE_IN_BIT_LOCATION
        = (static_cast<uint64_t>(Gmma_mode) & ((1u << GMMA_DESCRIPTOR_MODE_BITS) - 1)) << GMMA_DESCRIPTOR_MODE_SHIFT;

    // stride byte offset, bit 32-45, 4LSB not included
    // each column is always of 128 byte. 8 columns always.
    // divide by 16 since the 4 LSB is not included
    static constexpr uint64_t STRIDE_BYTE_OFFSET = BYTES_PER_LEADING_DIM
        * ((Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B)     ? 8
                : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                 : 2)
        / 16;
    // shift 32 bit
    static constexpr uint64_t STRIDE_BYTE_OFFSET_IN_BIT_LOCATION = STRIDE_BYTE_OFFSET << 32;

    // leading byte offset, bit 16-29, 4LSB not included
    // each column is still 128 byte.
    // divide by 16 since the 4 LSB is not included
    // for B matrix of TN, and the way we reshape the matrix, LEADING_BYTE_OFFSET is never non-zero
    // in the future with different GMMA shape, this might be needed
    static constexpr bool LEADING_BYTE_OFFSET_NEEDED
        = (((GMMA_N * BITS_PER_ELEMENT) / 8 > BYTES_PER_LEADING_DIM && Gmma_trans == Gmma_descriptor_transpose::TRANS)
              || GMMA_K == 64)
        ? true
        : false;

    // the leading byte offset if needed 4LSB not included
    static constexpr uint64_t LEADING_BYTE_OFFSET = GMMA_K == 64
        ? Cta_tile::N * 32 / 16
        : (BYTES_PER_LEADING_DIM * ((Gmma_trans == Gmma_descriptor_transpose::TRANS) ? Cta_tile::K : Cta_tile::N) / 16);
    // shift 16 bit
    static constexpr uint64_t LEADING_BYTE_OFFSET_IN_BIT_LOCATION
        = LEADING_BYTE_OFFSET_NEEDED ? LEADING_BYTE_OFFSET << 16 : 0;

    // ctor
    inline __device__ Gmma_descriptor_b()
    {
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] = 0;
        }

// set bit 62-63 to 1 for SWIZZLE_128B format
// set bit 62-63 to 2 for SWIZZLE_64B format
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] |= DESCRIPTOR_MODE_IN_BIT_LOCATION;
        }

// stride byte offset, bit 32-45, 4LSB not included
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] |= STRIDE_BYTE_OFFSET_IN_BIT_LOCATION;
        }

        // leading byte offset, bit 16-29, 4LSB not included
        if (LEADING_BYTE_OFFSET_NEEDED)
        {
#pragma unroll
            for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
            {
                desc[desc_idx] |= LEADING_BYTE_OFFSET_IN_BIT_LOCATION;
            }
        }
    }

    // update the descriptor based on smem address. Should be called once from prologue.
    inline __device__ void set_smem_pointer(uint32_t smem_nvvm_pointer)
    {
        // uint64_t smem_address_bit = reinterpret_cast<uint64_t>(smem);
        // uint32_t smem_nvvm_pointer = get_smem_pointer(smem);
        uint64_t smem_address_bit = static_cast<uint64_t>(smem_nvvm_pointer);

        // set base offset, bit 49-61
        uint64_t offset = (smem_address_bit / BYTES_PER_LEADING_DIM)
            % ((Gmma_mode == Gmma_descriptor_mode::SWIZZLE_128B)     ? 8
                    : Gmma_mode == Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                     : 2);
        uint64_t offset_in_bit_location = offset << 49;
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            desc[desc_idx] |= offset_in_bit_location;
        }

// start_address, bit 0-13, 4LSB not included(so grab bit 4-17)
// the only bits that is different for each desc of the same obj
#pragma unroll
        for (int desc_idx = 0; desc_idx < NUM_DESCRIPTORS; ++desc_idx)
        {
            // for fp16, desc_idx_in_128B should range from 0 to 3
            int desc_idx_in_128B = desc_idx % NUM_DESCRIPTORS_PER_128B_IN_K;
            int desc_idx_over_128B = desc_idx / NUM_DESCRIPTORS_PER_128B_IN_K;

            uint64_t smem_address_bit_in_bit_location
                = (smem_address_bit + ((GMMA_K * BITS_PER_ELEMENT) / 8) * desc_idx_in_128B
                      + Cta_tile::N * BYTES_PER_LEADING_DIM * desc_idx_over_128B)
                << 46;
            smem_address_bit_in_bit_location = smem_address_bit_in_bit_location >> 50;
            desc[desc_idx] |= smem_address_bit_in_bit_location;
        }
    }

    // get a single desc from the desc group.
    inline __device__ uint64_t get_descriptor(int desc_idx) const
    {
        // if(threadIdx.x == 128)
        //    printf("desc[0] = 0x%lx\n", desc[0]);
        //__syncwarp();
        return desc[(Gmma_vector_size == Gmma_descriptor_size::ALL) ? desc_idx : 0];
    }

    // get the max descriptor for desc[0]
    inline __device__ uint64_t get_max_descriptor_0() const
    {
        return max_desc_0;
    }

    // set a single desc from the desc group.
    inline __device__ void set_descriptor(int desc_idx, uint64_t single_desc)
    {
        desc[(Gmma_vector_size == Gmma_descriptor_size::ALL) ? desc_idx : 0] = single_desc;
    }

    // set the max descriptor for desc[0]. Should be called once from prologue.
    // Should be called with set_smem_pointer()
    // This value is needed to "loop back" to the first LDGSTS buffer when appropriate.
    inline __device__ void set_max_descriptor_0(int mem_offset_no_4LSB)
    {
        max_desc_0 = desc[0] + mem_offset_no_4LSB;
    }

    // for desc group where all desc all allocated,
    // increment_single_descriptor() will do nothing.
    inline __device__ void increment_single_descriptor(bool last_of_kblock)
    {
        // update smem start address, which is in lower 32bits.
        int2& tmp = reinterpret_cast<int2&>(desc[0]);
        if (last_of_kblock == true)
        {
            tmp.x -= BYTES_DESC_INC_BOUNDARY_NO_4LSB;
        }
        else
        {
            tmp.x += BYTES_PER_DESC_NO_4LSB;
        }
    }

    template <int BYTE_OFFSET>
    inline __device__ void increment_single_descriptor()
    {
        int2& tmp = reinterpret_cast<int2&>(desc[0]);
        tmp.x += (BYTE_OFFSET >> 4);
    }

    // for desc group where all desc all allocated,
    // increment_single_descriptor() will do nothing.
    inline __device__ void increment_single_descriptor(bool last_of_kblock, bool switch_kblock)
    {
        // update smem start address, which is in lower 32bits.
        int2& tmp = reinterpret_cast<int2&>(desc[0]);
        if (RESET_SMEM)
        {
            if (switch_kblock)
            {
                tmp.x -= BYTES_PER_DESC_NO_4LSB;
                tmp.x += RESET_BYTES_NO_4LSB;
            }
            else
            {
                if (last_of_kblock == true)
                {
                    tmp.x -= BYTES_PER_DESC_NO_4LSB;
                    tmp.x -= RESET_BYTES_NO_4LSB;
                }
                else
                {
                    tmp.x += BYTES_PER_DESC_NO_4LSB;
                }
            }
        }
        else
        {
            if (last_of_kblock == true)
            {
                tmp.x -= BYTES_DESC_INC_BOUNDARY_NO_4LSB;
            }
            else
            {
                tmp.x += BYTES_PER_DESC_NO_4LSB;
            }
        }
    }

private:
    // the descriptors, each of 64 bit
    uint64_t desc[NUM_DESCRIPTORS];
    // the max desc for desc_idx = 0
    uint64_t max_desc_0;
};

} // namespace fmha
