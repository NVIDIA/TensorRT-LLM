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

#include <fmha/fragment.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/smem_tile_v.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
/// @brief Interface to Smem tiles for a operator
//  HGMMA
//
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Gmma_fusion_mode
{
    NO_FUSION,
    BN_APPLY
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace wip
{

template <typename Traits, typename Cta_tile, typename Layout, int BUFFERS_PER_TILE = 1,
    fmha::Gmma_descriptor_mode desc_mode = fmha::Gmma_descriptor_mode::SWIZZLE_128B, bool GMMA_A_RF = Traits::GMMA_A_RF,
    // if USE_LDGSTS is false, TMA will be used.
    bool USE_LDGSTS = true>
struct Smem_tile_hopper_a
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Layout, int BUFFERS_PER_TILE = 1,
    fmha::Gmma_descriptor_mode desc_mode = fmha::Gmma_descriptor_mode::SWIZZLE_128B, bool GMMA_B_RF = Traits::GMMA_B_RF,
    // if USE_LDGSTS is false, TMA will be used.
    bool USE_LDGSTS = true>
struct Smem_tile_hopper_b
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Col Major. For GMMA, A is from SMEM directly.
// Not implemented, since it is not really needed at the moment.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_col_a
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major. For GMMA, A is from SMEM directly.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_row_a
{
    // Currently Interleaved Mode is not implemented.
    static_assert(
        desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, SWIZZLE_NONE Mode is not implemented.\n");

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of desc within a gmma group (kblock limited).
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor.
    using Gmma_descriptor = fmha::Gmma_descriptor_a<fmha::Gmma_descriptor_transpose::NOTRANS, desc_mode, Cta_tile,
        Traits::BITS_PER_ELEMENT_A, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    using Cta_tile_gmma = Cta_tile;

    // the size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A
    };

    // the size of bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size in bytes of a single LDGSTS/STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The number of elements per LDGSTS/STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B
    // and SWIZZLE_64B format.
    enum
    {
        BYTES_PER_ROW = 128
    };

    // the number of rows per one row of K due the the limitation of leading dim size.
    enum
    {
        NUM_ROWS_PER_K = (Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1) / BYTES_PER_ROW
    };

    static_assert(desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_64B || (Cta_tile::K * BYTES_PER_ELEMENT) == 64,
        "swizzle_64B row_a is valid if kblock=32\n");

    // Number of SMEM rows.
    enum
    {
        NUM_ROWS
        = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? (Cta_tile::M * NUM_ROWS_PER_K) : (Cta_tile::M / 2)
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW
    };

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer.
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc.
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // The number of buffers.
    enum
    {
        BUFFERS_PER_TILE = BUFFERS_PER_TILE_
    };

    // The size in bytes of total buffers.
    enum
    {
        BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER
    };

    // The number of threads needed to store a row
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS
    };

    // The number of rows written with a single STS.
    enum
    {
        ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // for swizzle_128B the xor factor is 8
    enum
    {
        ROWS_PER_XOR_PATTERN = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? 8 : 4
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum
    {
        GMMA_GROUP_SMEM_DISTANCE
        = Mma_tile::M_PER_GMMA_GROUP / (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2) * BYTES_PER_ROW
    };

    // The number of STS per row.
    enum
    {
        STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS
    };

    // For Hopper, STS_PER_ROW should be 1 (at least for now.)
    static_assert(STS_PER_ROW == 1, "");

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_row_a(char* smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem))
    {

        int smem_write_row = tidx / THREADS_PER_ROW;
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN;
        int smem_write_col = 0;

        if (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B)
        {
            smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;
        }
        else if (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B)
        {
            smem_write_col = (tidx % (THREADS_PER_ROW / 2))
                ^ smem_write_xor + ((tidx % THREADS_PER_ROW) / (THREADS_PER_ROW / 2)) * 4;
        }

        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW + smem_write_col * BYTES_PER_STS;

        // That code is expected to trigger the utilization of the URF by the compiler.
        this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template <int N>
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N])
    {
#pragma unroll
        for (int ii = 0; ii < N; ++ii)
        {
            // Decompose the STS into row/col.
            int row = ii / STS_PER_ROW;
            // Assemble the offset.
            int offset = smem_write_offset_ + row * ROWS_PER_STS * BYTES_PER_ROW;
            // Assemble the final pointer :)
            ptrs[ii] = smem_ + offset + smem_write_buffer_;
        }
    }

    // Store the tile in the shared memory.
    template <int N, int M>
    inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t (&preds)[M], uint64_t = 0)
    {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers<N>(smem_ptrs);
        ldgsts<N, M>(smem_ptrs, gmem_ptrs, preds);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer()
    {
        if (BUFFERS_PER_TILE > 1)
        {
            this->smem_write_offset_ += (smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY)
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER;
        }
    }

    inline __device__ void move_next_write_buffer(int) {}

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
    // The buffer base offset for write.
    int smem_write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Col Major. For GMMA, B is from SMEM directly.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_col_b
{

    // Currently Interleaved Mode is not implemented.
    static_assert(
        desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, Interleaved Mode is not implemented.\n");

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = fmha::Gmma_descriptor_b<fmha::Gmma_descriptor_transpose::NOTRANS, desc_mode, Cta_tile,
        Traits::BITS_PER_ELEMENT_B, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    using Cta_tile_gmma = Cta_tile;

    // the size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B
    };

    // the size of bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size in bytes of a single LDGSTS/STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The number of elements per LDGSTS/STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum
    {
        BYTES_PER_COLUMN = 128
    };

    static_assert(desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_64B || (Cta_tile::K * BYTES_PER_ELEMENT) == 64,
        "swizzle_64B col_b is valid if kblock=32\n");

    // the number of columns per one column of K due the the limitation of leading dim size
    enum
    {
        NUM_COLS_PER_K = (Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1) / BYTES_PER_COLUMN
    };

    // Number of SMEM columns.
    enum
    {
        NUM_COLUMNS
        = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? Cta_tile::N * NUM_COLS_PER_K : Cta_tile::N / 2
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = NUM_COLUMNS * BYTES_PER_COLUMN
    };

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc.
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // The number of buffers.
    enum
    {
        BUFFERS_PER_TILE = BUFFERS_PER_TILE_
    };

    // The size in bytes of total buffers.
    enum
    {
        BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER
    };

    // The number of threads needed to store a column.
    enum
    {
        THREADS_PER_COLUMN = BYTES_PER_COLUMN / BYTES_PER_STS
    };

    // The number of columns written with a single STS.
    enum
    {
        COLUMNS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN
    };

    // for swizzle_128B the xor factor is 8.
    enum
    {
        COLUMNS_PER_XOR_PATTERN = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? 8 : 4
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum
    {
        GMMA_GROUP_SMEM_DISTANCE = Mma_tile::N_PER_GMMA_GROUP
            / (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2) * BYTES_PER_COLUMN
    };

    // The number of STS per column.
    enum
    {
        STS_PER_COLUMN = BYTES_PER_COLUMN / THREADS_PER_COLUMN / BYTES_PER_STS
    };

    // For Hopper, STS_PER_COLUMN should be 1 (at least for now.)
    static_assert(STS_PER_COLUMN == 1, "");

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_col_b(char* smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem))
    {
        int smem_write_col = tidx / THREADS_PER_COLUMN;
        int smem_write_xor = smem_write_col % COLUMNS_PER_XOR_PATTERN;
        int smem_write_row = 0;

        if (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B)
        {
            smem_write_row = (tidx % THREADS_PER_COLUMN) ^ smem_write_xor;
        }
        else if (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B)
        {
            smem_write_row = (tidx % (THREADS_PER_COLUMN / 2))
                ^ smem_write_xor + ((tidx % THREADS_PER_COLUMN) / (THREADS_PER_COLUMN / 2)) * 4;
        }

        this->smem_write_offset_ = smem_write_col * BYTES_PER_COLUMN + smem_write_row * BYTES_PER_STS;
        // That code is expected to trigger the utilization of the URF by the compiler.
        this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template <int N>
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N])
    {
#pragma unroll
        for (int ii = 0; ii < N; ++ii)
        {
            // Decompose the STS into row/col.
            int col = ii / STS_PER_COLUMN;
            // Assemble the offset.
            int offset = smem_write_offset_ + col * COLUMNS_PER_STS * BYTES_PER_COLUMN;
            // Assemble the final pointer :)
            ptrs[ii] = smem_ + offset + smem_write_buffer_;
        }
    }

    // Store the tile in the shared memory.
    template <int N, int M>
    inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t (&preds)[M], uint64_t = 0)
    {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers<N>(smem_ptrs);
        ldgsts<N, M>(smem_ptrs, gmem_ptrs, preds);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer()
    {
        // if( BUFFERS_PER_TILE > 1 ) {
        //     this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
        //                                     ? -BYTES_PER_TILE_INC_BOUNDARY
        //                                     : BYTES_PER_BUFFER;
        // }
    }

    inline __device__ void move_next_write_buffer(int) {}

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
    // The buffer base offset for write.
    int smem_write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Row Major. For GMMA, B is from SMEM directly.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_row_b
{

    // Currently Interleaved Mode is not implemented.
    static_assert(
        desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, Interleaved Mode is not implemented.\n");

    // For SWIZZLE_64B, row b is not needed/implemented
    static_assert(desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_64B,
        "Currently, for SWIZZLE_64B mode, row_b is not needed/implemented. \n");

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = fmha::Gmma_descriptor_b<fmha::Gmma_descriptor_transpose::TRANS, desc_mode, Cta_tile,
        Traits::BITS_PER_ELEMENT_B, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    using Cta_tile_gmma = Cta_tile;

    // the size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B
    };

    // the size of bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size in bytes of a single LDGSTS/STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The number of elements per LDGSTS/STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum
    {
        BYTES_PER_ROW = 128
    };

    // the number of rows per one row of N due the the limitation of leading dim size
    enum
    {
        NUM_ROWS_PER_N = (Cta_tile::N * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1) / BYTES_PER_ROW
    };

    // the number of rows per one row of N_PER_GMMA_GROUP
    enum
    {
        NUM_ROWS_PER_GMMA_GROUP_N = (Mma_tile::N_PER_GMMA_GROUP * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1) / BYTES_PER_ROW
    };

    // Number of SMEM rows
    enum
    {
        NUM_ROWS = Cta_tile::K * NUM_ROWS_PER_N
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW
    };

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // The number of buffers.
    enum
    {
        BUFFERS_PER_TILE = BUFFERS_PER_TILE_
    };

    // The size in bytes of total buffers.
    enum
    {
        BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER
    };

    // The number of threads needed to store a row
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS
    };

    // The number of rows written with a single STS.
    enum
    {
        ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // for swizzle_128B the xor factor is 8
    enum
    {
        ROWS_PER_XOR_PATTERN = 8
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum
    {
        GMMA_GROUP_SMEM_DISTANCE = Mma_tile::K_PER_GMMA_GROUP * NUM_ROWS_PER_GMMA_GROUP_N * BYTES_PER_ROW
    };

    // The number of STS per ROW.
    enum
    {
        STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS
    };

    // For Hopper, STS_PER_ROW should be 1 (at least for now.)
    static_assert(STS_PER_ROW == 1, "");

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_row_b(char* smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem))
    {
        int smem_write_row = tidx / THREADS_PER_ROW;
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN;
        int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;
        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW + smem_write_col * BYTES_PER_STS;
        // That code is expected to trigger the utilization of the URF by the compiler.
        this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template <int N>
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N])
    {
#pragma unroll
        for (int ii = 0; ii < N; ++ii)
        {
            // Decompose the STS into row/col.
            int row = ii / STS_PER_ROW;
            // Assemble the offset.
            int offset = smem_write_offset_ + row * ROWS_PER_STS * BYTES_PER_ROW;

            // Assemble the final pointer :)
            ptrs[ii] = smem_ + offset + smem_write_buffer_;
        }
    }

    template <int N, int M>
    inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t (&preds)[M], uint64_t = 0)
    {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers<N>(smem_ptrs);
        ldgsts<N, M>(smem_ptrs, gmem_ptrs, preds);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer()
    {
        // if( BUFFERS_PER_TILE > 1 ) {
        //     this->smem_write_offset_ += ( smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY )
        //                                     ? -BYTES_PER_TILE_INC_BOUNDARY
        //                                     : BYTES_PER_BUFFER;
        // }
    }

    inline __device__ void move_next_write_buffer(int) {}

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    // The shared memory pointer.
    uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    int smem_read_buffer_;
    // The buffer base offset for write.
    int smem_write_buffer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized Interface
// LDGSTS smem tiles.
////////////////////////////////////////////////////////////////////////////////////////////////////
// A Col Major, A coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits, Cta_tile, fmha::Col, BUFFERS_PER_TILE_, desc_mode, false,
    true // use ldgsts
    > : public Smem_tile_hopper_gmma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_col_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    // comment the implementation out as a mark that this is not supported, yet.
    // inline __device__ Smem_tile_hopper_a( char *smem, int tidx ) : Base( smem, tidx ) {
    //}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major, A coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits, Cta_tile, fmha::Row, BUFFERS_PER_TILE_, desc_mode, false,
    true // use ldgsts
    > : public Smem_tile_hopper_gmma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Col Major, B coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, fmha::Col, BUFFERS_PER_TILE_, desc_mode, false,
    true // use ldgsts
    > : public Smem_tile_hopper_gmma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Row Major, B coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, fmha::Row, BUFFERS_PER_TILE_, desc_mode, false,
    true // use ldgsts
    > : public Smem_tile_hopper_gmma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b(char* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized Interface
// TMA smem tiles.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major. For GMMA, A is from SMEM directly.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_row_a
{
    // Currently Interleaved Mode is not implemented.
    static_assert(
        desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, SWIZZLE_NONE Mode is not implemented.\n");

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of desc within a gmma group (kblock limited).
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor.
    using Gmma_descriptor = fmha::Gmma_descriptor_a<fmha::Gmma_descriptor_transpose::NOTRANS, desc_mode, Cta_tile,
        Traits::BITS_PER_ELEMENT_A, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    using Cta_tile_gmma = Cta_tile;

    // the size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_A
    };

    // the size of bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size in bytes of a single LDGSTS/STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The number of elements per LDGSTS/STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B
    // and SWIZZLE_64B format.
    enum
    {
        BYTES_PER_ROW = 128
    };

    // the number of rows per one row of K due the the limitation of leading dim size.
    enum
    {
        NUM_ROWS_PER_K = (Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1) / BYTES_PER_ROW
    };

    static_assert(desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_64B || (Cta_tile::K * BYTES_PER_ELEMENT) == 64,
        "swizzle_64B row_a is valid if kblock=32\n");

    // Number of SMEM rows.
    enum
    {
        NUM_ROWS
        = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? (Cta_tile::M * NUM_ROWS_PER_K) : (Cta_tile::M / 2)
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW
    };

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer.
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc.
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // The number of buffers.
    enum
    {
        BUFFERS_PER_TILE = BUFFERS_PER_TILE_
    };

    // The size in bytes of total buffers.
    enum
    {
        BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER
    };

    // The number of threads needed to store a row
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS
    };

    // The number of rows written with a single STS.
    enum
    {
        ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // for swizzle_128B the xor factor is 8
    enum
    {
        ROWS_PER_XOR_PATTERN = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? 8 : 4
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum
    {
        GMMA_GROUP_SMEM_DISTANCE
        = Mma_tile::M_PER_GMMA_GROUP / (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2) * BYTES_PER_ROW
    };

    // The number of STS per row.
    enum
    {
        STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS
    };

    // For Hopper, STS_PER_ROW should be 1 (at least for now.)
    static_assert(STS_PER_ROW == 1, "");

    // Each smem barrier is of 8 bytes
    enum
    {
        BYTES_PER_SMEM_BARRIER = 8
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY_SMEM_BARRIER = BYTES_PER_SMEM_BARRIER * BUFFERS_PER_TILE - BYTES_PER_SMEM_BARRIER
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_tma_row_a(char* smem, char* smem_barrier)
        : smem_(__nvvm_get_smem_pointer(smem))
        , smem_barrier_(__nvvm_get_smem_pointer(smem_barrier))
        , smem_write_offset_(0)
        , smem_barrier_offset_(0)
    {
    }

    // Move the write offset to next buffer.
    // Also move the smem_barrier.
    inline __device__ void move_next_write_buffer()
    {
        if (BUFFERS_PER_TILE > 1)
        {
            this->smem_write_offset_ += (smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY)
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER;
        }

        // also update the smem_barrier.
        if (BUFFERS_PER_TILE > 1)
        {
            this->smem_barrier_offset_ += (smem_barrier_offset_ >= BYTES_PER_TILE_INC_BOUNDARY_SMEM_BARRIER)
                ? -BYTES_PER_TILE_INC_BOUNDARY_SMEM_BARRIER
                : BYTES_PER_SMEM_BARRIER;
        }
    }

    inline __device__ void move_next_write_buffer(int) {}

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    template <int DIM, cudaTmaDescType DESC_TYPE>
    inline __device__ void store(cudaTmaDesc const* p_desc, int32_t const (&coord)[DIM], uint16_t filter_offsets = 0,
        uint16_t mcast_cta_mask = 0)
    {

        fmha::utmaldg<DIM, DESC_TYPE, false>(
            p_desc, smem_ + smem_write_offset_, smem_barrier_ + smem_barrier_offset_, coord);
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The barrier in smem.
    uint32_t smem_barrier_;
    // The write offset.
    int smem_write_offset_;
    // The smem barrier offset
    int smem_barrier_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Col Major. For GMMA, B is from SMEM directly.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_col_b
{

    // Currently Interleaved Mode is not implemented.
    static_assert(
        desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, Interleaved Mode is not implemented.\n");

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = fmha::Gmma_descriptor_b<fmha::Gmma_descriptor_transpose::NOTRANS, desc_mode, Cta_tile,
        Traits::BITS_PER_ELEMENT_B, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    using Cta_tile_gmma = Cta_tile;

    // the size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B
    };

    // the size of bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size in bytes of a single LDGSTS/STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The number of elements per LDGSTS/STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum
    {
        BYTES_PER_COLUMN = 128
    };

    static_assert(desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_64B || (Cta_tile::K * BYTES_PER_ELEMENT) == 64,
        "swizzle_64B col_b is valid if kblock=32\n");

    // the number of columns per one column of K due the the limitation of leading dim size
    enum
    {
        NUM_COLS_PER_K = (Cta_tile::K * BYTES_PER_ELEMENT + BYTES_PER_COLUMN - 1) / BYTES_PER_COLUMN
    };

    // Number of SMEM columns.
    enum
    {
        NUM_COLUMNS
        = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? Cta_tile::N * NUM_COLS_PER_K : Cta_tile::N / 2
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = NUM_COLUMNS * BYTES_PER_COLUMN
    };

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc.
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // The number of buffers.
    enum
    {
        BUFFERS_PER_TILE = BUFFERS_PER_TILE_
    };

    // The size in bytes of total buffers.
    enum
    {
        BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER
    };

    // The number of threads needed to store a column.
    enum
    {
        THREADS_PER_COLUMN = BYTES_PER_COLUMN / BYTES_PER_STS
    };

    // The number of columns written with a single STS.
    enum
    {
        COLUMNS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_COLUMN
    };

    // for swizzle_128B the xor factor is 8.
    enum
    {
        COLUMNS_PER_XOR_PATTERN = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B) ? 8 : 4
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum
    {
        GMMA_GROUP_SMEM_DISTANCE = Mma_tile::N_PER_GMMA_GROUP
            / (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B ? 1 : 2) * BYTES_PER_COLUMN
    };

    // The number of STS per column.
    enum
    {
        STS_PER_COLUMN = BYTES_PER_COLUMN / THREADS_PER_COLUMN / BYTES_PER_STS
    };

    // For Hopper, STS_PER_COLUMN should be 1 (at least for now.)
    static_assert(STS_PER_COLUMN == 1, "");

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_tma_col_b(char* smem, char* smem_barrier)
        : smem_(__nvvm_get_smem_pointer(smem))
        , smem_barrier_(__nvvm_get_smem_pointer(smem_barrier))
    {
    }

    // Move the write offset to next buffer.
    // Not implemented as it is not needed currently.
    inline __device__ void move_next_write_buffer() {}

    inline __device__ void move_next_write_buffer(int) {}

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    template <int DIM, cudaTmaDescType DESC_TYPE>
    inline __device__ void store(cudaTmaDesc const* p_desc, int32_t const (&coord)[DIM], uint16_t filter_offsets = 0,
        uint16_t mcast_cta_mask = 0)
    {

        fmha::utmaldg<DIM, DESC_TYPE, false>(p_desc, smem_, smem_barrier_, coord);
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The barrier in smem.
    uint32_t smem_barrier_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Row Major. For GMMA, B is from SMEM directly.
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_gmma_tma_row_b
{

    // Currently Interleaved Mode is not implemented.
    static_assert(
        desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_NONE, "Currently, Interleaved Mode is not implemented.\n");

    // For SWIZZLE_64B, row b is not needed/implemented
    static_assert(desc_mode != fmha::Gmma_descriptor_mode::SWIZZLE_64B,
        "Currently, for SWIZZLE_64B mode, row_b is not needed/implemented. \n");

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;

    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = fmha::Gmma_descriptor_b<fmha::Gmma_descriptor_transpose::TRANS, desc_mode, Cta_tile,
        Traits::BITS_PER_ELEMENT_B, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    using Cta_tile_gmma = Cta_tile;

    // the size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = Traits::BITS_PER_ELEMENT_B
    };

    // the size of bytes of each element.
    enum
    {
        BYTES_PER_ELEMENT = BITS_PER_ELEMENT / 8
    };

    // The size in bytes of a single LDGSTS/STS.
    enum
    {
        BYTES_PER_STS = 16
    };

    // The number of elements per LDGSTS/STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // SMEM layout for GMMA has a leading dim of exact 128 Byte, at least for SWIZZLE_128B and
    // SWIZZLE_64B format
    enum
    {
        BYTES_PER_ROW = 128
    };

    // the number of rows per one row of N due the the limitation of leading dim size
    enum
    {
        NUM_ROWS_PER_N = (Cta_tile::N * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1) / BYTES_PER_ROW
    };

    // the number of rows per one row of N_PER_GMMA_GROUP
    enum
    {
        NUM_ROWS_PER_GMMA_GROUP_N = (Mma_tile::N_PER_GMMA_GROUP * BYTES_PER_ELEMENT + BYTES_PER_ROW - 1) / BYTES_PER_ROW
    };

    // Number of SMEM rows
    enum
    {
        NUM_ROWS = Cta_tile::K * NUM_ROWS_PER_N
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = NUM_ROWS * BYTES_PER_ROW
    };

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // The number of buffers.
    enum
    {
        BUFFERS_PER_TILE = BUFFERS_PER_TILE_
    };

    // The size in bytes of total buffers.
    enum
    {
        BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE
    };

    // The boundary for smem_read_offset and smem_write_offset increment.
    enum
    {
        BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER
    };

    // The number of threads needed to store a row
    enum
    {
        THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STS
    };

    // The number of rows written with a single STS.
    enum
    {
        ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // for swizzle_128B the xor factor is 8
    enum
    {
        ROWS_PER_XOR_PATTERN = 8
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    enum
    {
        GMMA_GROUP_SMEM_DISTANCE = Mma_tile::K_PER_GMMA_GROUP * NUM_ROWS_PER_GMMA_GROUP_N * BYTES_PER_ROW
    };

    // The number of STS per ROW.
    enum
    {
        STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS
    };

    // For Hopper, STS_PER_ROW should be 1 (at least for now.)
    static_assert(STS_PER_ROW == 1, "");

    // Ctor.
    inline __device__ Smem_tile_hopper_gmma_tma_row_b(char* smem, char* smem_barrier)
        : smem_(__nvvm_get_smem_pointer(smem))
        , smem_barrier_(__nvvm_get_smem_pointer(smem_barrier))
    {
    }

    // Move the write offset to next buffer.
    // Not implemented since it is not needed at the moment.
    inline __device__ void move_next_write_buffer() {}

    inline __device__ void move_next_write_buffer(int) {}

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    template <int DIM, cudaTmaDescType DESC_TYPE>
    inline __device__ void store(cudaTmaDesc const* p_desc, int32_t const (&coord)[DIM], uint16_t filter_offsets = 0,
        uint16_t mcast_cta_mask = 0)
    {

        fmha::utmaldg<DIM, DESC_TYPE, false>(p_desc, smem_, smem_barrier_, coord);
    }

    // The shared memory pointer.
    uint32_t smem_;
    // The barrier in smem.
    uint32_t smem_barrier_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A Row Major, A coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_a<Traits, Cta_tile, fmha::Row, BUFFERS_PER_TILE_, desc_mode, false,
    false // will be use tma
    > : public Smem_tile_hopper_gmma_tma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_row_a<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_a(char* smem, char* smem_barrier)
        : Base(smem, smem_barrier)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Col Major, B coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, fmha::Col, BUFFERS_PER_TILE_, desc_mode, false,
    false // will be use tma
    > : public Smem_tile_hopper_gmma_tma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_col_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b(char* smem, char* smem_barrier)
        : Base(smem, smem_barrier)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// B Row Major, B coming from SMEM
template <typename Traits, typename Cta_tile, int BUFFERS_PER_TILE_, fmha::Gmma_descriptor_mode desc_mode>
struct Smem_tile_hopper_b<Traits, Cta_tile, fmha::Row, BUFFERS_PER_TILE_, desc_mode, false,
    false // will be use tma
    > : public Smem_tile_hopper_gmma_tma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>
{

    // The base class.
    using Base = Smem_tile_hopper_gmma_tma_row_b<Traits, Cta_tile, BUFFERS_PER_TILE_, desc_mode>;

    // Ctor.
    inline __device__ Smem_tile_hopper_b(char* smem, char* smem_barrier)
        : Base(smem, smem_barrier)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace wip

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The description of the tile computed by this CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The number of bytes per STS.
    int BYTES_PER_STS_,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // Whether to use TMA.
    bool USE_TMA,
    // Whether A is coming for RF.
    bool GMMA_A_RF = Traits_::GMMA_A_RF>
struct Smem_tile_hopper_a
    : public fmha::Smem_tile_without_skews<Cta_tile_, Layout_::COL ? Cta_tile_::K : Cta_tile_::M,
          Layout_::COL ? Cta_tile_::M : Cta_tile_::K, Traits_::BITS_PER_ELEMENT_A, BYTES_PER_STS_, BUFFERS_PER_TILE_, 0,
          (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B         ? 8
                  : desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                         : 2),
          1, true, USE_TMA, 128 * 8 / Traits_::BITS_PER_ELEMENT_A>
{
    using Traits = Traits_;
    using Cta_tile = Cta_tile_;
    // The base class.
    using Base = fmha::Smem_tile_without_skews<Cta_tile, Layout_::COL ? Cta_tile::K : Cta_tile::M,
        Layout_::COL ? Cta_tile::M : Cta_tile::K, Traits::BITS_PER_ELEMENT_A, BYTES_PER_STS_, BUFFERS_PER_TILE_, 0,
        (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B         ? 8
                : desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                       : 2),
        1, true, USE_TMA, 128 * 8 / Traits::BITS_PER_ELEMENT_A>;

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The layout
    using Layout = Layout_;
    // The fragment.
    using Fragment = fmha::Fragment_a<Traits, Layout>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;
    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = fmha::Gmma_descriptor_a<
        Layout::COL ? fmha::Gmma_descriptor_transpose::TRANS : fmha::Gmma_descriptor_transpose::NOTRANS, desc_mode,
        Cta_tile, Traits::BITS_PER_ELEMENT_A, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    // the number of columns per one column of M_PER_GMMA_GROUP
    enum
    {
        NUM_COLS_PER_GMMA_GROUP_M
        = (Mma_tile::M_PER_GMMA_GROUP * Base::BITS_PER_ELEMENT / 8 + Base::BYTES_PER_ROW - 1) / Base::BYTES_PER_ROW
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock
    static constexpr int GMMA_GROUP_SMEM_DISTANCE = Layout::COL
        ? (Mma_tile::K_PER_GMMA_GROUP * NUM_COLS_PER_GMMA_GROUP_M * Base::BYTES_PER_ROW * Cta_tile::WARP_GROUP_M)
        : (Mma_tile::M_PER_GMMA_GROUP * Cta_tile::WARP_GROUP_M
            / (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B       ? 1
                    : desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B ? 2
                                                                           : 4)
            * Base::BYTES_PER_ROW);

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = Base::BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    // set the scale and bias smem pointer
    inline __device__ void set_scale_bias_smem_ptr(char* scale_bias_smem_ptr, int tidx, int k) {}

    // Load from shared memory.
    template <typename Layout_b>
    inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki)
    {
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    // Overload set needs to be replicated for compatibility
    inline __device__ void move_next_read_buffer(int N) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The description of the tile computed by this CTA.
    typename Cta_tile_,
    // The layout of the tile.
    typename Layout_,
    // The number of bytes per STS.
    int BYTES_PER_STS_,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // USe TMA or not,
    bool USE_TMA>
struct Smem_tile_hopper_b : public fmha::Smem_tile_without_skews<Cta_tile_,
                                Layout_::COL ? Cta_tile_::N : Cta_tile_::K, // ROWS
                                Layout_::COL ? Cta_tile_::K : Cta_tile_::N, // COLS
                                Traits_::BITS_PER_ELEMENT_B, BYTES_PER_STS_, BUFFERS_PER_TILE_,
                                0,                                          // LDS_FAST_PATH
                                // Determine ROWS_PER_XOR_PATTERN from the swizzle mode:
                                (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B         ? 8
                                        : desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                                               : /* 32B or NONE */ 2),
                                1,                                    // COLS_PER_XOR_PATTERN
                                true,                                 // USE_PREDICATES
                                USE_TMA,
                                128 * 8 / Traits_::BITS_PER_ELEMENT_B // LEAD_DIM_ELEMENTS
                                >
{
    using Traits = Traits_;
    using Cta_tile = Cta_tile_;
    // The base class.
    using Base = fmha::Smem_tile_without_skews<Cta_tile, Layout_::COL ? Cta_tile::N : Cta_tile::K,
        Layout_::COL ? Cta_tile::K : Cta_tile::N, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS_, BUFFERS_PER_TILE_, 0,
        (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B         ? 8
                : desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B ? 4
                                                                       : 2),
        1, true, USE_TMA, 128 * 8 / Traits::BITS_PER_ELEMENT_B>;

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The layout
    using Layout = Layout_;
    // The fragment.
    using Fragment = fmha::Fragment_b<Traits, Layout>;

    // The number of desc within a gmma group (kblock limited)
    static constexpr fmha::Gmma_descriptor_size GMMA_DESC_SIZE_PER_GROUP = fmha::Gmma_descriptor_size::ONE;
    // The SWIZZLE_128B descriptor
    using Gmma_descriptor = fmha::Gmma_descriptor_b<
        Layout::COL ? fmha::Gmma_descriptor_transpose::NOTRANS : fmha::Gmma_descriptor_transpose::TRANS, desc_mode,
        Cta_tile, Traits::BITS_PER_ELEMENT_B, Traits::GMMA_M, Traits::GMMA_N, Traits::GMMA_K, GMMA_DESC_SIZE_PER_GROUP>;

    // the number of rows per one row of N_PER_GMMA_GROUP
    enum
    {
        NUM_ROWS_PER_GMMA_GROUP_N
        = (Mma_tile::N_PER_GMMA_GROUP * Base::BITS_PER_ELEMENT / 8 + Base::BYTES_PER_ROW - 1) / Base::BYTES_PER_ROW
    };

    // The distance in byte between different GMMA groups (might need multiple due to cta tile size)
    // each GMMA group is of size GMMA_M x GMMA_N x Kblock

    // The dimension that we split.
    // Add buffers when we have multiple buffers for split head dimensions.
    // Split-d smem view (2 split D, and 3 buffers): d0, d0, d0, d1, d1, d1.
    static constexpr int GMMA_GROUP_SPLIT_DIM
        = Layout::COL ? Mma_tile::N_PER_GMMA_GROUP : (Mma_tile::K_PER_GMMA_GROUP * BUFFERS_PER_TILE_);

    // The split factor.
    static constexpr int GMMA_GROUP_SPLIT_FACTOR = (desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_128B ? 1
            : desc_mode == fmha::Gmma_descriptor_mode::SWIZZLE_64B                                        ? 2
                                                                                                          : 4);

    // Make sure the dimension that we split is a multiple of the split factor.
    static_assert(GMMA_GROUP_SPLIT_DIM % GMMA_GROUP_SPLIT_FACTOR == 0);

    // The distance between two "groups" in shared memory.
    static constexpr int GMMA_GROUP_SMEM_DISTANCE
        = GMMA_GROUP_SPLIT_DIM / GMMA_GROUP_SPLIT_FACTOR * Base::BYTES_PER_ROW;

    // the size of one buffer in bytes in shared memory, without the 4 LSB.
    // this is needed to increment the GMMA desc to the next buffer
    enum
    {
        BYTES_PER_BUFFER_NO_4LSB = Base::BYTES_PER_BUFFER / 16
    };

    // this is needed to decrement GMMA desc
    enum
    {
        BYTES_PER_BUFFER_INC_BOUNDARY_NO_4LSB = BYTES_PER_BUFFER_NO_4LSB * BUFFERS_PER_TILE_ - BYTES_PER_BUFFER_NO_4LSB
    };

    // Ctor.
    inline __device__ Smem_tile_hopper_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
        warp_id_ = tidx / 32;
        lane_id_ = tidx % 32;

        // each pair of warps transposes 8x8 in place
        // each warp responsible for diagonal 4x4s
        // calculate index in 8x8 block
        block_row_ = lane_id_ / 4;
        block_col_ = (lane_id_ % 4) + ((warp_id_ % 2) ^ (block_row_ / 4)) * 4;

        // diagonal 4x4s will 2x conflict for SWIZZLE_32B
        // 1 warp per 8x8, 2 4x8 load+store
        if (Traits::GMMA_N == 8)
        {
            block_row_ = lane_id_ / 8;
            block_col_ = lane_id_ % 8;
        }

        // offset when all 4 warps participate in transpose
        block_col_offset_ = (warp_id_ / 2) * 8;
    }

    int warp_id_, lane_id_;
    int block_row_, block_col_, block_col_offset_;

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {}

    // Load from smem, do something (e.g. transpose), then store back to smem
    inline __device__ void load_and_store(int ki)
    {
        /*
        using B_type = typename Traits::B_type;

        // TODO: move these to B_RF smem tiles

        // 8 channel per group fp16 fprop/dgrad with 64x16x16 gmma
        // move 8x8 OOB zeros to right diagonal, 8x8 in-bounds weights on left diagonal
        if (Cta_tile::N_PER_GROUP == 8 && Traits::GMMA_N == 16
                && Traits::BITS_PER_ELEMENT_B == 16) {
            // just need to swap 2 cores within a single SWIZZLE_32B, one of which is just zero
            // 1 LDSM.M88.1
            if (warp_id_ == 0) {
                int smem_row_offset = ki * 4 * 128 + 2 * 128; // 4 rows per 16x16, swap the bottom 8x16
                int lds_block_idx = lane_id_ * 2; // ldsm.m88.1 only uses first 8 threads for address
                int lds_smem_idx = lds_block_idx ^ (lane_id_ / 4);

                uint32_t data;
                uint32_t lds_smem_ptr = this->smem_ + this->smem_read_buffer_
                                        + smem_row_offset
                                        + lds_smem_idx * 16;
                fmha::ldsm(data, lds_smem_ptr);

                __syncwarp();

                // move values to adjacent core
                fmha::stsm(lds_smem_ptr ^ 16, data);

                // set zeros at previous core
                fmha::stsm(lds_smem_ptr, static_cast<uint32_t>(0));
            }
        }

        // 4 channel per group tf32 fprop with 64x8x8 gmma
        // move 4x4 in-bounds weights on left diagonal, OOB zeros everywhere else
        if (Cta_tile::N_PER_GROUP == 4 && Traits::GMMA_N == 8
                && Layout::COL && Traits::BITS_PER_ELEMENT_B == 32) {
            // just need to swap the bottom 4x8, 1 elt per thread for 1 warp
            // 1 lds/sts.32 per thread
            if (warp_id_ == 0) {
                int smem_row_offset = ki * Base::ROWS_PER_XOR_PATTERN * 128 + 128;
                int lds_smem_idx = lane_id_;
                uint32_t lds_ptr = this->smem_ + this->smem_read_buffer_
                                    + smem_row_offset
                                    + lds_smem_idx * sizeof(B_type);
                uint32_t data;
                lds(data, lds_ptr);

                __syncwarp();

                sts(lds_ptr ^ 16, data);
            }
        }

        // partial transpose of 8xN_PER_GROUP operand for tf32 grouped dgrad
        // todo: revise this for tf32 grouped wgrad, move to partial specialization
        static constexpr bool IS_TF32_GROUPED_DGRAD =
            (Cta_tile::GROUPS_N > 1 && Cta_tile::GROUPS_K > 1 || Cta_tile::N_PER_GROUP == 32)
                && Layout::ROW && Traits::BITS_PER_ELEMENT_B == 32;
        if (IS_TF32_GROUPED_DGRAD) {
            static constexpr int XOR_SCALE = 16 / sizeof(B_type); // 16B swizzle over 4B elements
            static constexpr int ROWS_PER_128B = kDivUp( 128, Traits::GMMA_N * sizeof(B_type) );

            if (Traits::GMMA_N == 8) {
                if (warp_id_ == 0) {

                int smem_row_offset = ki * Base::ROWS_PER_XOR_PATTERN * 128;
                uint32_t data[2];

                #pragma unroll
                for (int ii = 0; ii < 2; ii++) {
                    // get index in row-major 8x8
                    int lds_block_row = block_row_ + ii * 4;
                    int lds_block_col = block_col_;
                    int lds_block_idx = lds_block_row * 8 + lds_block_col;

                    // swizzle
                    int lds_xor_factor = (lds_block_row / ROWS_PER_128B) * XOR_SCALE;
                    int lds_smem_idx = lds_block_idx ^ lds_xor_factor;

                    // Load from smem
                    uint32_t lds_ptr = this->smem_ + this->smem_read_buffer_
                                        + smem_row_offset
                                        + lds_smem_idx * sizeof(B_type);
                    lds(data[ii], lds_ptr);
                }

                __syncwarp();

                #pragma unroll
                for (int ii = 0; ii < 2; ii++) {
                    // get index in col-major 8x8
                    int sts_block_row = block_col_;
                    int sts_block_col = block_row_ + ii * 4;
                    if (Cta_tile::N_PER_GROUP == 4 && ii == 1) {
                        // place 4x4 weights on diagonal for 4-channel tf32 group dgrad
                        sts_block_row ^= 4;
                    }
                    int sts_block_idx = sts_block_row * 8 + sts_block_col;

                    // swizzle
                    int sts_xor_factor = (sts_block_row / ROWS_PER_128B) * XOR_SCALE;
                    int sts_smem_idx = sts_block_idx ^ sts_xor_factor;

                    // store to smem
                    uint32_t sts_ptr = this->smem_ + this->smem_read_buffer_
                                        + smem_row_offset
                                        + sts_smem_idx * sizeof(B_type);
                    sts(sts_ptr, data[ii]);
                }

                } // warp_id == 0
            } else {
                // loop over 8x16 blocks
                #pragma unroll
                for (int ii = 0; ii < kDivUp(Cta_tile::N_PER_GROUP, 16); ii++) {
                    int smem_row_offset = ki * Base::ROWS_PER_XOR_PATTERN * 128;

                    // get index in row-major 8xN_PER_GROUP
                    int lds_block_row = block_row_;
                    int lds_block_col = block_col_ + block_col_offset_ + ii * 16;
                    int lds_block_idx = lds_block_row * Cta_tile::N_PER_GROUP
                                        + lds_block_col;

                    // swizzle
                    int lds_xor_factor = (lds_block_row / ROWS_PER_128B) * XOR_SCALE;
                    int lds_smem_idx = lds_block_idx ^ lds_xor_factor;

                    // Load from smem
                    uint32_t lds_ptr = this->smem_ + this->smem_read_buffer_
                                        + smem_row_offset
                                        + lds_smem_idx * sizeof(B_type);
                    uint32_t data;
                    lds(data, lds_ptr);

                    __syncwarp();

                    // get index in row-major 8xN_PER_GROUP with 8x8 in-place transposes
                    int sts_block_row = block_col_;
                    int sts_block_col = block_row_ + block_col_offset_ + ii * 16;
                    int sts_block_idx = sts_block_row * Cta_tile::N_PER_GROUP
                                        + sts_block_col;

                    // swizzle
                    int sts_xor_factor = (sts_block_row / ROWS_PER_128B) * XOR_SCALE;
                    int sts_smem_idx = sts_block_idx ^ sts_xor_factor;

                    // store to smem
                    uint32_t sts_ptr = this->smem_ + this->smem_read_buffer_
                                        + smem_row_offset
                                        + sts_smem_idx * sizeof(B_type);
                    sts(sts_ptr, data);
                }
            }
        }

        // make sure sts are visible to gmma
        fence_view_async_shared();
        */
    }

    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer() {}

    // Move the read offset to next buffer.
    inline __device__ void move_next_read_buffer(int buffer_id)
    {
        this->smem_read_buffer_ = buffer_id * Base::BYTES_PER_BUFFER;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template < // GMMA instruction shape in M dim
    int GMMA_M,
    // GMMA instruction shape in N dim
    int GMMA_N,
    // GMMA instruction shape in K dim
    int GMMA_K,
    // GMMA A operand coming from RF?
    bool GMMA_A_RF,
    // GMMA B operand coming from RF?
    bool GMMA_B_RF,
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // Use TMA or not,
    bool USE_TMA, int BUFFERS_PER_TILE>
struct Smem_tile_v<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    BUFFERS_PER_TILE, desc_mode, USE_TMA>
    : public fmha::Smem_tile_hopper_b<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile, fmha::Row,
          16, // BYTES_PER_STS
          BUFFERS_PER_TILE, desc_mode, USE_TMA>
{

    static constexpr bool TRANSPOSE = false;

    using Cta_tile_gmma = Cta_tile;

    using Base = fmha::Smem_tile_hopper_b<fmha::Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile, fmha::Row,
        16, // BYTES_PER_STS
        BUFFERS_PER_TILE, desc_mode, USE_TMA>;

    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void transpose_tile(int)
    {
        // Transpose is fused into HGMMA.
    }

    inline __device__ void transpose_tile(int, uint32_t, uint32_t)
    {
        // Transpose is fused into HGMMA.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template < // GMMA instruction shape in M dim
    int GMMA_M,
    // GMMA instruction shape in N dim
    int GMMA_N,
    // GMMA instruction shape in K dim
    int GMMA_K,
    // GMMA A operand coming from RF?
    bool GMMA_A_RF,
    // GMMA B operand coming from RF?
    bool GMMA_B_RF,
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // Use TMA or not,
    bool USE_TMA, int BUFFERS_PER_TILE>
struct Smem_tile_v<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    BUFFERS_PER_TILE, desc_mode, USE_TMA>
    : public fmha::Smem_tile_hopper_b<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile, fmha::Row,
          16,               // BYTES_PER_STS
          BUFFERS_PER_TILE, // BUFFERS_PER_TILE,
          desc_mode, USE_TMA>
{

    static constexpr bool TRANSPOSE = false;

    using Cta_tile_gmma = Cta_tile;

    using Base = fmha::Smem_tile_hopper_b<fmha::Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile, fmha::Row,
        16,               // BYTES_PER_STS
        BUFFERS_PER_TILE, // BUFFERS_PER_TILE,
        desc_mode, USE_TMA>;

    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void transpose_tile(int)
    {
        // Transpose is fused into HGMMA.
    }

    inline __device__ void transpose_tile(int, uint32_t, uint32_t)
    {
        // Transpose is fused into HGMMA.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template < // GMMA instruction shape in M dim
    int GMMA_M,
    // GMMA instruction shape in N dim
    int GMMA_N,
    // GMMA instruction shape in K dim
    int GMMA_K,
    // GMMA A operand coming from RF?
    bool GMMA_A_RF,
    // GMMA B operand coming from RF?
    bool GMMA_B_RF,
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // Use TMA or not,
    bool USE_TMA, int BUFFERS_PER_TILE>
struct Smem_tile_v<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    BUFFERS_PER_TILE, desc_mode, USE_TMA>
    : public fmha::Smem_tile_hopper_b<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile, fmha::Row,
          16,               // BYTES_PER_STS
          BUFFERS_PER_TILE, // BUFFERS_PER_TILE,
          desc_mode, USE_TMA>
{

    static constexpr bool TRANSPOSE = false;

    using Cta_tile_gmma = Cta_tile;

    using Base = fmha::Smem_tile_hopper_b<fmha::Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
        Cta_tile, fmha::Row,
        16,               // BYTES_PER_STS
        BUFFERS_PER_TILE, // BUFFERS_PER_TILE,
        desc_mode, USE_TMA>;

    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }

    inline __device__ void transpose_tile(int)
    {
        // Transpose is fused into HGMMA.
    }

    inline __device__ void transpose_tile(int, uint32_t, uint32_t)
    {
        // Transpose is fused into HGMMA.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int L, int UNROLL_N = 1>
struct Transposer
{
};

template <typename Traits, typename Cta_tile, int UNROLL_N>
struct Transposer<Traits, Cta_tile, 128, UNROLL_N>
{

    static_assert(Cta_tile::K % 128 == 0);

    enum
    {
        WARPS_M = Cta_tile::WARPS_M,
        WARPS_N = Cta_tile::WARPS_N,
        WARPS_K = Cta_tile::WARPS_K,
    };

    enum
    {
        WARPS_4x1x1 = (WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1),
        WARPS_4x1x2 = (WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 2),
    };

    enum
    {
        BYTES_PER_LDS = 16
    };

    enum
    {
        BYTES_PER_ROW = 128
    };

    // D=64 and 4 warps.
    // Per warp we load 32 rows x 16 columns with LDSM.Tx4, 128 rows per CTA.
    enum
    {
        S = Cta_tile::K >= 128 ? 128 : Cta_tile::K
    }; // The sequence length.

    enum
    {
        D = Cta_tile::N >= 128 ? 128 : Cta_tile::N
    }; // The head dimension.

    // static_assert(S % 128 == 0);
    static_assert(WARPS_4x1x1 || WARPS_4x1x2);
    static_assert(D % (BYTES_PER_LDS * WARPS_K) == 0);

    enum
    {
        ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING = 128
    }; // LDSMx4

    enum
    {
        ROW_PACKING = BYTES_PER_ROW / (D * sizeof(typename Traits::B_type))
    };

    enum
    {
        ROWS_PER_LDSM_PER_CTA = ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING / ROW_PACKING
    };

    enum
    {
        ROWS_PER_XOR_PATTERN = fmha::Rows_per_xor_pattern_ampere_b<Traits, S>::VALUE
    };

    static_assert(ROWS_PER_XOR_PATTERN == 8);

    // The number of loads in K dimension.
    enum
    {
        K = S / ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING
    };

    // static_assert(K * ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING == S);
    // static_assert(K == 3);
    //  The number of loads in the D dimension.
    enum
    {
        N = D / (BYTES_PER_LDS * WARPS_K)
    }; // 16 bytes per load

    static_assert(N * BYTES_PER_LDS * WARPS_K == D);

    uint4 regs_[UNROLL_N][K];

    uint32_t read_offset_;
    uint32_t write_offset_;
    uint32_t smem_read_loc_;
    uint32_t smem_write_loc_;

    inline __device__ Transposer(int tidx)
    {

        int read_row, read_col;

        if (WARPS_4x1x1 && N == 8)
        { // D=128, 1 warp  in N
            read_row = (tidx & 0x7f);
            read_col = (tidx & 0x07);
        }
        else if (WARPS_4x1x1 && N == 4)
        { // D=64, 1 warp  in N
            read_row = (tidx & 0xe0) / 2 + (tidx & 0x1e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        }
        else if (WARPS_4x1x1 && N == 2)
        { // D=32, 1 warp  in N
            read_row = (tidx & 0x60) / 4 + (tidx & 0x1c) / 4;
            read_col = (tidx & 0x03) * 2;
            read_col ^= (read_row & 0x01);
        }
        else if (WARPS_4x1x2 && N == 4)
        { // D=128, 2 warps in N
            read_row = (tidx & 0x7f);
            read_col = (tidx & 0x07);
            // For two warpgroups we do two steps in N at once.
            read_col ^= (tidx & 0x80) / 128;
        }
        else if (WARPS_4x1x2 && N == 2)
        { // D=64, 2 warps in N
            read_row = (tidx & 0x60) / 2 + (tidx & 0x1e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
            // For two warpgroups we do two steps in N at once.
            read_col ^= (tidx & 0x80) / 128;
        }
        else if (WARPS_4x1x2 && N == 1)
        { // D=32, 2 warps  in N
            read_row = (tidx & 0x60) / 4 + (tidx & 0x1c) / 4;
            read_col = (tidx & 0x03) * 2;
            read_col ^= (read_row & 0x01);
            // For two warpgroups we do two steps in N at once.
            read_col ^= (tidx & 0x80) / 128;
        }
        else
        {
            assert(false);
        }

        read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

        int write_row, write_col;
        if (WARPS_4x1x1)
        { // swizzle_128byte
            write_row = (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x60) / 16 + (tidx & 0x08) / 8;
        }
        else if (WARPS_4x1x2)
        {
            // Same as above, with second warp group writing next 16 rows.
            write_row = (tidx & 0x80) / 8 + (tidx & 0x10) / 2 + (tidx & 0x07);
            write_col = (tidx & 0x60) / 16 + (tidx & 0x08) / 8;
        }
        else
        {
            assert(false);
        }

        write_col ^= (write_row & 0x07);

        write_offset_ = write_row * BYTES_PER_ROW + write_col * BYTES_PER_LDS;
    }

    inline __device__ void transpose(int tidx, uint32_t smem)
    {
        transpose_<true>(tidx, smem, smem);
    }

    template <bool SYNC>
    inline __device__ void transpose_(uint32_t smem_src, uint32_t smem_dst)
    {
#pragma unroll
        for (int n_begin = 0; n_begin < N; n_begin += UNROLL_N)
        {
            transpose_ldmatrix(n_begin, smem_src);
            transpose_stmatrix<SYNC>(n_begin, smem_dst);
        }
    }

    inline __device__ void transpose_ldmatrix(int n_begin, uint32_t smem_src)
    {
        static_assert(N % UNROLL_N == 0, "");

        uint4 tmp[UNROLL_N][K];
        if (n_begin == 0)
        {
            smem_read_loc_ = smem_src + read_offset_;
        }

#pragma unroll
        for (int ni = n_begin; ni < n_begin + UNROLL_N; ni++)
        {
            int const nii = ni - n_begin;
#pragma unroll
            for (int ki = 0; ki < K; ki++)
            { // 2
                fmha::ldsmt(tmp[nii][ki], smem_read_loc_ + ki * ROWS_PER_LDSM_PER_CTA * BYTES_PER_ROW);
            }

            if (WARPS_4x1x1 && N == 4)
            { // D=64, 1 warp  in N
                smem_read_loc_ ^= (ni % 2 == 0 ? 1 : 3) * 16;
            }
            else if (WARPS_4x1x1 && N == 2)
            { // D=32, 1 warp  in N
                smem_read_loc_ ^= 16;
            }
            else if (WARPS_4x1x2 && N == 2)
            { // D=64, 2 warps in N
                smem_read_loc_ ^= 32;
            }
            else if (WARPS_4x1x2 && N == 4)
            { // D=128, 2 warps in N
                smem_read_loc_ ^= (ni % 2 == 0 ? 1 : 3) * 32;
            }
            else if (WARPS_4x1x1 && N == 8)
            { // D=128, 1 warp  in N
                smem_read_loc_ ^= ((ni % 4 == 3) ? 7 : (ni % 2 == 1 ? 3 : 1)) * 16;
            }
            else if (N != 1)
            {
                assert(false);
            }
        }

#pragma unroll
        for (int ni = n_begin; ni < n_begin + UNROLL_N; ni++)
        {
            int const nii = ni - n_begin;
#pragma unroll
            for (int ki = 0; ki < K; ki++)
            {
                fmha::swizzle_rows(regs_[nii][ki].x, regs_[nii][ki].z, tmp[nii][ki].x, tmp[nii][ki].y); // PRMT 0+1
                fmha::swizzle_rows(regs_[nii][ki].y, regs_[nii][ki].w, tmp[nii][ki].z, tmp[nii][ki].w); // PRMT 2+3
            }
        }
    }

    template <bool SYNC>
    inline __device__ void transpose_stmatrix(int n_begin, uint32_t smem_dst)
    {

        // After LDSM.Tx4 registers hold 2x2 elts:
        // [00, 01]
        // [10, 11]
        // With row offsets
        // x: + 0
        // y: + 8
        // z: +16 (g)
        // w: +24 (o)
        //
        // After PRMT 0, the :
        // [00, 01] [80, 81] => x: [00, 10, 80, 90], i.e. col 0
        // [10, 11] [90, 91] => z: [01, 11, 81, 91], i.e. col 1
        //
        // [g0, g1] [o0, o1] => y: [g0, h0, o0, p0], i.e. col 0
        // [h0, h1] [p0, p1] => w: [g1, h1, o1, p1], i.e. col 1
        //
        // Therefore, when looking at the transpose, quad q holds cols 2 * q + [0, 1], i.e.
        // - quad 0 holds cols 0, 1
        // - quad 1 holds cols 2, 3
        // - etc.
        //
        // This fits with the accumulator layout, since N strides in steps of 8 per thread.

        if (SYNC)
        {                    // needed if src and dst are the same.
            __syncthreads(); // LDSM.T done. We should now have a D x S tile in registers. SMEM can be written.
        }

        if (n_begin == 0)
        {
            smem_write_loc_ = smem_dst + write_offset_;
        }

#pragma unroll
        for (int ni = n_begin; ni < n_begin + UNROLL_N; ni++)
        {
            int const nii = ni - n_begin;
#pragma unroll
            for (int ki = 0; ki < K; ki++)
            {
                fmha::stsm(smem_write_loc_ + ki * BYTES_PER_ROW * D, regs_[nii][ki]);
            }
            if (WARPS_4x1x1)
            { // D=64, 1 warp in N.
                smem_write_loc_ += 16 * BYTES_PER_ROW;
            }
            else if (WARPS_4x1x2)
            { // D=64, 2 warps in N.
                smem_write_loc_ += 32 * BYTES_PER_ROW;
            }
            else
            {
                assert(false);
            }
        }
    }
};

template <typename Traits, typename Cta_tile, int UNROLL_N>
struct Transposer<Traits, Cta_tile, 64, UNROLL_N>
{

    static_assert(Cta_tile::K % 64 == 0);

    enum
    {
        WARPS_M = Cta_tile::WARPS_M,
        WARPS_N = Cta_tile::WARPS_N,
        WARPS_K = Cta_tile::WARPS_K,
    };

    enum
    {
        WARPS_4x1x1 = (WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1),
        WARPS_4x1x2 = (WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 2),
    };

    enum
    {
        BYTES_PER_LDS = 16
    };

    // D=64 and 4 warps.
    // Per warp we load 32 rows x 16 columns with LDSM.Tx4, 128 rows per CTA.
    enum
    {
        S = Cta_tile::K >= 128 ? 128 : Cta_tile::K
    }; // The sequence length.

    enum
    {
        D = Cta_tile::N >= 128 ? 128 : Cta_tile::N
    }; // The head dimension.

    static_assert(S % 64 == 0);
    static_assert(WARPS_4x1x1);
    static_assert(D % 32 == 0);

    static_assert(S == 64 && D == 128);

    // Two warps in S dim.
    enum
    {
        ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING = 64
    }; // LDSMx4

    enum
    {
        BYTES_PER_ROW = 128
    };

    enum
    {
        ROW_PACKING = Div_up<BYTES_PER_ROW, D * sizeof(typename Traits::B_type)>::VALUE
    };

    enum
    {
        ROWS_PER_LDSM_PER_CTA = ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING / ROW_PACKING
    }; // due to row_packing

    // The number of loads in K dimension.
    enum
    {
        K = S / ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING
    };

    // The number of loads in the D dimension. Use two warps in D dim.
    enum
    {
        N = D / 32
    };

    uint4 regs_[UNROLL_N][K];

    uint32_t read_offset_;
    uint32_t write_offset_;
    uint32_t smem_read_loc_;
    uint32_t smem_write_loc_;

    inline __device__ Transposer(int tidx)
    {
        int read_row, read_col;

        if (WARPS_4x1x1 && N == 1)
        { // D=32, 2 warps  in N
            read_row = (tidx & 0x20) / 4 + (tidx & 0x1c) / 4;
            read_col = (tidx & 0x03) * 2;
            read_col ^= (read_row & 0x01);
            read_col ^= ((tidx & 0x40) / 64);
        }
        else if (WARPS_4x1x1 && N == 2)
        { // D=64, 2 warps  in N
            read_row = (tidx & 0x20) / 2 + (tidx & 0x1e) / 2;
            read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
            read_col ^= ((tidx & 0x40) / 64);
        }
        else if (WARPS_4x1x1 && N == 4)
        { // D=128, 2 warps  in N
            read_row = (tidx & 0x3f);
            read_col = (tidx & 0x07);
            read_col ^= ((tidx & 0x40) / 64);
        }
        else
        {
            assert(false);
        }

        read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

        // static_assert(ROWS_PER_LDSM_PER_CTA == 32);
        // constexpr int ROWS_PER_XOR_PATTERN = 4;
        // constexpr int ROWS_PER_XOR_PATTERN = fmha::Rows_per_xor_pattern_ampere_b<Traits, S>::VALUE;

        int row, col;
        if (WARPS_4x1x1)
        {
            row = (tidx & 0x40) / 4 + (tidx & 0x10) / 2 + (tidx & 0x07);
            col = (tidx & 0x20) / 16 + (tidx & 0x08) / 8;
            col = col + (row % 2) * 4;
            row = row / 2;
            col = col ^ (row % 4);
        }
        else
        {
            assert(false);
        }
        write_offset_ = row * BYTES_PER_ROW + col * BYTES_PER_LDS;
    };

    inline __device__ void transpose(int tidx, uint32_t smem)
    {
        transpose_<true>(tidx, smem, smem);
    }

    template <bool SYNC>
    inline __device__ void transpose_(uint32_t smem_src, uint32_t smem_dst)
    {
#pragma unroll
        for (int n_begin = 0; n_begin < N; n_begin += UNROLL_N)
        {
            transpose_ldmatrix(n_begin, smem_src);
            transpose_stmatrix<SYNC>(n_begin, smem_dst);
        }
    }

    inline __device__ void transpose_ldmatrix(int n_begin, uint32_t smem_src)
    {
        static_assert(N % UNROLL_N == 0, "");

        uint4 tmp[UNROLL_N][K];
        if (n_begin == 0)
        {
            smem_read_loc_ = smem_src + read_offset_;
        }
#pragma unroll
        for (int ni = n_begin; ni < n_begin + UNROLL_N; ni++)
        {
#pragma unroll
            for (int ki = 0; ki < K; ki++)
            {
                int const nii = ni - n_begin;
                fmha::ldsmt(tmp[ni][ki], smem_read_loc_ + ki * ROWS_PER_LDSM_PER_CTA * BYTES_PER_ROW);
            }

            if (WARPS_4x1x1 && N == 2)
            { // D=64, 2 warps in N
                smem_read_loc_ ^= 32;
            }
            else if (WARPS_4x1x1 && N == 4)
            { // D=128, 2 warps in N
                smem_read_loc_ ^= (ni % 2 == 1 ? 3 * 32 : 32);
            }
            else if (N != 1)
            {
                assert(false);
            }
        }

#pragma unroll
        for (int ni = n_begin; ni < n_begin + UNROLL_N; ni++)
        {
            int const nii = ni - n_begin;
#pragma unroll
            for (int ki = 0; ki < K; ki++)
            {
                fmha::swizzle_rows(regs_[nii][ki].x, regs_[nii][ki].z, tmp[nii][ki].x, tmp[nii][ki].y); // PRMT 0+1
                fmha::swizzle_rows(regs_[nii][ki].y, regs_[nii][ki].w, tmp[nii][ki].z, tmp[nii][ki].w); // PRMT 2+3
            }
        }
    }

    template <bool SYNC>
    inline __device__ void transpose_stmatrix(int n_begin, uint32_t smem_dst)
    {

        // After LDSM.Tx4 registers hold 2x2 elts:
        // [00, 01]
        // [10, 11]
        // With row offsets
        // x: + 0
        // y: + 8
        // z: +16 (g)
        // w: +24 (o)
        //
        // After PRMT 0, the :
        // [00, 01] [80, 81] => x: [00, 10, 80, 90], i.e. col 0
        // [10, 11] [90, 91] => z: [01, 11, 81, 91], i.e. col 1
        //
        // [g0, g1] [o0, o1] => y: [g0, h0, o0, p0], i.e. col 0
        // [h0, h1] [p0, p1] => w: [g1, h1, o1, p1], i.e. col 1
        //
        // Therefore, when looking at the transpose, quad q holds cols 2 * q + [0, 1], i.e.
        // - quad 0 holds cols 0, 1
        // - quad 1 holds cols 2, 3
        // - etc.
        //
        // This fits with the accumulator layout, since N strides in steps of 8 per thread.

        if (SYNC)
        {
            __syncthreads(); // LDSM.T done. We should now have a D x S tile in registers. SMEM can be written.
        }

        if (n_begin == 0)
        {
            smem_write_loc_ = smem_dst + write_offset_;
        }

#pragma unroll
        for (int ni = n_begin; ni < n_begin + UNROLL_N; ni++)
        {
            int const nii = ni - n_begin;
#pragma unroll
            for (int ki = 0; ki < K; ki++)
            {
                fmha::stsm(smem_write_loc_ + ki * BYTES_PER_ROW * D / 2, regs_[nii][ki]);
            }
            if (WARPS_4x1x1)
            { // D=64, 1 warp in N.
                smem_write_loc_ += 16 * BYTES_PER_ROW;
            }
            else
            {
                assert(false);
            }
        }
    }
};

template <
    // The instruction traits.
    typename Traits,
    // The Cta_tile.
    typename Cta_tile,
    // The number of buffers.
    int BUFFERS_PER_TILE,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // USe TMA or not,
    bool USE_TMA>
struct Smem_tile_v_gmma
{

    static_assert(sizeof(typename Traits::B_type) == 1);

    // K is the sequence length dimension (128 for GMMA)
    enum
    {
        K_ = Cta_tile::K % 128 == 0 ? 128 : 64
    };

    static_assert(Cta_tile::K % K_ == 0);

    // static_assert(Cta_tile::N == 128);
    // static_assert(K_ == 128);
    // static_assert(BUFFERS_PER_TILE == 2);

    using Cta_tile_gmma_ = typename Traits::template Cta_tile<Cta_tile::M, Cta_tile::N, K_, Cta_tile::WARP_GROUP_M,
        Cta_tile::WARP_GROUP_N, Cta_tile::WARP_GROUP_K>;

    // TODO Swizzle_32B?
    static constexpr fmha::Gmma_descriptor_mode GMMA_DESC_MODE_V
        = Cta_tile_gmma_::K * sizeof(typename Traits::B_type) >= 128 ? fmha::Gmma_descriptor_mode::SWIZZLE_128B
                                                                     : fmha::Gmma_descriptor_mode::SWIZZLE_64B;

    static_assert((Cta_tile::K % 128 == 0 && GMMA_DESC_MODE_V == fmha::Gmma_descriptor_mode::SWIZZLE_128B)
        || (Cta_tile::K % 64 == 0 && GMMA_DESC_MODE_V == fmha::Gmma_descriptor_mode::SWIZZLE_64B));

    enum
    {
        NUM_KGROUPS = Cta_tile::K / Cta_tile_gmma_::K
    };

    static_assert(NUM_KGROUPS * Cta_tile_gmma_::K == Cta_tile::K);

    enum
    {
        BYTES_PER_STS = 16
    };

    // The compute tile only requires static information from Smem_tile_v and accesses SMEM directly through GMMA.
    // Hence, we declare a SxD column major matrix in SMEM and have to make sure at runtime that the data is transposed.
    // Note that for K > 128, we are using two buffers per tile, which we have to fill accordingly.
    using Base_ = fmha::Smem_tile_hopper_b<Traits, Cta_tile_gmma_, fmha::Col, BYTES_PER_STS,
        BUFFERS_PER_TILE * NUM_KGROUPS, GMMA_DESC_MODE_V, USE_TMA>;

    // Split D or not, which influences the GMMA_GROUP_SMEM_DISTANCE, and BYTES_PER_BUFFER_NO_4LSB.
    // Split-d smem view (2 split D, and 3 buffers): d0, d0, d0, d1, d1, d1.
    // The group distance would be number_of_buffers * buffer_size.
    // The buffer size is the size for split-d.
    static constexpr size_t GMMA_GROUP_SMEM_DISTANCE = Base_::GMMA_GROUP_SMEM_DISTANCE * BUFFERS_PER_TILE;
    static constexpr size_t BYTES_PER_BUFFER_NO_4LSB = Base_::BYTES_PER_BUFFER_NO_4LSB;

    using Gmma_descriptor = typename Base_::Gmma_descriptor;

    struct Base : public Base_
    {
        using Transposer = Transposer<Traits, Cta_tile, K_>;
        static_assert(USE_TMA == false);
        static constexpr bool TRANSPOSE = true;

        enum
        {
            NUM_KGROUPS = Cta_tile::K / Cta_tile_gmma_::K
        };

        enum
        {
            ROWS_PER_XOR_PATTERN = fmha::Rows_per_xor_pattern_ampere_b<Traits, Cta_tile::N>::VALUE
        };

        using Descriptor = typename Base_::Gmma_descriptor;

        // Delegate all the stores to the Row-Major Smem_tile.
        using Store_delegate = Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 8, BYTES_PER_STS, 1, 0,
            ROWS_PER_XOR_PATTERN, 1>;

        using Store_type = typename Store_delegate::Store_type;

        enum
        {
            S = Cta_tile::K
        };

        // static_assert(Descriptor::BYTES_PER_LEADING_DIM == 128);
        // static_assert(Descriptor::STRIDE_BYTE_OFFSET == K_ * 8 / 16);  // 128 * 8 / 16
        // static_assert(Descriptor::TRANS_MODE == fmha::Gmma_descriptor_transpose::NOTRANS);
        // static_assert(Base::BYTES_PER_TILE == S * 64);
        // static_assert(!Descriptor::LEADING_BYTE_OFFSET_NEEDED);
        // static_assert(Descriptor::LEADING_BYTE_OFFSET == 128 * 64 / 16);
        // static_assert(Descriptor::BYTES_PER_DESC_NO_4LSB == 32 * 1 / 16);
        // static_assert(Descriptor::BYTES_DESC_INC_BOUNDARY_NO_4LSB == (K_ / 32 - 1) * 2);
        // static_assert(Base::BYTES_PER_BUFFER_NO_4LSB == K_ * 64 / 16);
        // static_assert(Base::GMMA_GROUP_SMEM_DISTANCE == 128 * 128 * 2);
        // static_assert(Base::BYTES_PER_BUFFER_NO_4LSB == 128 * 128);

        // static_assert(Store_delegate::N_WITH_PADDING == 64);
        // static_assert(Store_delegate::ROWS_PER_XOR_PATTERN == 4);
        // static_assert(Store_delegate::BYTES_PER_ROW_BEFORE_PACKING == 64);
        // static_assert(Store_delegate::ROWS == S / 2);
        // static_assert(Store_delegate::BYTES_PER_ROW == 128);

        // Number of rows a warp loads per LDSMx4
        enum
        {
            ROWS_PER_LDSM = 4 * 8
        };

        enum
        {
            ROWS_PER_LDSM_PER_CTA = ROWS_PER_LDSM * Cta_tile::WARPS_M
        };

        static_assert(Cta_tile::WARPS_M == 4);

        enum
        {
            LDSMS = Cta_tile::K / ROWS_PER_LDSM_PER_CTA
        };

        // TODO we're assigning all rows loaded by a warp group (128 per CTA) to the K dimension.
        // This only works for K a multiple of 128.
        // For S=192, we want 3 blocks of 64xD.
        // static_assert(LDSMS * ROWS_PER_LDSM_PER_CTA == Cta_tile::K);

        static_assert(LDSMS == S / 128);

        enum
        {
            BYTES_PER_LDS = 16
        };

        enum
        {
            BYTES_PER_ROW = Store_delegate::BYTES_PER_ROW
        };

        enum
        {
            WARPS_M = Cta_tile::WARPS_M,
            WARPS_N = Cta_tile::WARPS_N,
            WARPS_K = Cta_tile::WARPS_K,
        };

        enum
        {
            WARPS_4x1x1 = (WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 1),
            WARPS_4x1x2 = (WARPS_M == 4 && WARPS_N == 1 && WARPS_K == 2),
        };

        inline __device__ Base(void* smem, int tidx)
            : Base_(smem, tidx)
            , delegate(smem, tidx)
            , transposer(tidx)
        {
        }

        // Store to the tile in shared memory.
        template <int N>
        inline __device__ void store(Store_type const (&data)[N])
        {
            uint32_t smem_ptrs[N];
            delegate.compute_store_pointers(smem_ptrs);
            sts(smem_ptrs, data);
        }

        // Store to the tile in shared memory.
        template <int N, int M>
        inline __device__ void store(Store_type const (&data)[N], uint32_t (&preds)[M])
        {
            uint32_t smem_ptrs[N];
            delegate.compute_store_pointers(smem_ptrs);
            sts(smem_ptrs, data, preds);
        }

        // Store to the tile in shared memory.
        template <int N>
        inline __device__ void store(Store_type const (&data)[N], uint32_t preds)
        {
            delegate.store(data, preds);
        }

        // Store to the tile in shared memory.
        template <int N, int M>
        inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t (&preds)[M])
        {
            uint32_t smem_ptrs[N];
            delegate.compute_store_pointers<N>(smem_ptrs);
            ldgsts<N, M>(smem_ptrs, gmem_ptrs, preds);
        }

        // Store to the tile in shared memory.
        template <int N>
        inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t preds, uint64_t = 0)
        {
            uint32_t tmp[1] = {preds};
            delegate.store(gmem_ptrs, tmp);
        }

        // Store to the tile in shared memory.
        template <int N>
        inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t preds)
        {
            uint32_t tmp[1] = {preds};
            delegate.store(gmem_ptrs, tmp);
        }

        // Initial offset (via tidx) has been moved to ctor
        inline __device__ void transpose_tile(int /* tidx */)
        {
            transposer.transpose(0, this->smem_);
        }

        template <int UNROLL_N = Transposer::N>
        inline __device__ void transpose_tile(uint32_t smem_src, uint32_t smem_dst)
        {
            transposer.template transpose_<false, UNROLL_N>(smem_src, smem_dst);
        }

        inline __device__ void transpose_tile_ldmatrix(int, uint32_t smem)
        {
            transposer.transpose_ldmatrix(0, smem);
        }

        inline __device__ void transpose_tile_stmatrix(int, uint32_t smem)
        {
            transposer.template transpose_stmatrix<false>(0, smem);
        }

        inline __device__ void transpose_tile_128(int tidx)
        {

            // D=64 and 4 warps.
            // Per warp we load 32 rows x 16 columns with LDSM.Tx4, 128 rows per CTA.
            constexpr int S = Cta_tile::K; // The sequence length.
            constexpr int D = Cta_tile::N; // The head dimension.
            // static_assert(S == 256);
            static_assert(D == 64);
            // static_assert(S % 128 == 0);
            static_assert(WARPS_4x1x1 || WARPS_4x1x2);
            static_assert(D % (16 * WARPS_K) == 0);

            constexpr int ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING = 128; // LDSMx4
            constexpr int BYTES_PER_ROW = 128;
            constexpr int ROW_PACKING = BYTES_PER_ROW / (D * sizeof(Traits::B_type));

            // The number of loads in K dimension.
            constexpr int K = S / ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING;
            // static_assert(K * ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING == S);
            // static_assert(K == 3);
            //  The number of loads in the D dimension.
            constexpr int N = D / (16 * WARPS_K);
            static_assert(N * 16 * WARPS_K == D);

            int read_row, read_col;

            if (WARPS_4x1x1 && N == 4)
            { // D=64, 1 warp  in N
                read_row = (tidx & 0xe0) / 2 + (tidx & 0x1e) / 2;
                read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
            }
            else if (WARPS_4x1x2 && N == 2)
            { // D=64, 2 warps in N
                read_row = (tidx & 0x60) / 2 + (tidx & 0x1e) / 2;
                read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
                // For two warpgroups we do two steps in N at once.
                read_col ^= (tidx & 0x80) / 128;
            }
            else
            {
                assert(false);
            }

            uint32_t offset = read_row * BYTES_PER_ROW + read_col * 16;

            constexpr int ROWS_PER_LDSM_PER_CTA
                = ROWS_PER_LDSM_PER_CTA_WITHOUT_PACKING / ROW_PACKING; // due to row_packing

            uint4 tmp[N][K];
            uint32_t smem_tmp = this->smem_; //__nvvm_get_smem_pointer(v_smem_) ;
            uint32_t smem_loc = smem_tmp + offset;

#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
#pragma unroll
                for (int ki = 0; ki < K; ki++)
                {
                    fmha::ldsmt(tmp[ni][ki], smem_loc + ki * ROWS_PER_LDSM_PER_CTA * BYTES_PER_ROW);
                }

                if (WARPS_4x1x1 && N == 4)
                { // D=64, 1 warp  in N
                    smem_loc ^= (ni % 2 == 0 ? 1 : 3) * 16;
                }
                else if (WARPS_4x1x2 && N == 2)
                { // D=64, 2 warps in N
                    smem_loc ^= 32;
                }
                else
                {
                    assert(false);
                }
            }

            uint4 regs[N][K];

#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
#pragma unroll
                for (int ki = 0; ki < K; ki++)
                {
                    fmha::swizzle_rows(regs[ni][ki].x, regs[ni][ki].z, tmp[ni][ki].x, tmp[ni][ki].y); // PRMT 0+1
                    fmha::swizzle_rows(regs[ni][ki].y, regs[ni][ki].w, tmp[ni][ki].z, tmp[ni][ki].w); // PRMT 2+3
                }
            }

            // After LDSM.Tx4 registers hold 2x2 elts:
            // [00, 01]
            // [10, 11]
            // With row offsets
            // x: + 0
            // y: + 8
            // z: +16 (g)
            // w: +24 (o)
            //
            // After PRMT 0, the :
            // [00, 01] [80, 81] => x: [00, 10, 80, 90], i.e. col 0
            // [10, 11] [90, 91] => z: [01, 11, 81, 91], i.e. col 1
            //
            // [g0, g1] [o0, o1] => y: [g0, h0, o0, p0], i.e. col 0
            // [h0, h1] [p0, p1] => w: [g1, h1, o1, p1], i.e. col 1
            //
            // Therefore, when looking at the transpose, quad q holds cols 2 * q + [0, 1], i.e.
            // - quad 0 holds cols 0, 1
            // - quad 1 holds cols 2, 3
            // - etc.
            //
            // This fits with the accumulator layout, since N strides in steps of 8 per thread.

            __syncthreads(); // LDSM.T done. We should now have a D x S tile in registers. SMEM can be written.
            constexpr int ROWS_PER_XOR_PATTERN = fmha::Rows_per_xor_pattern_ampere_b<Traits, S>::VALUE;
            static_assert(ROWS_PER_XOR_PATTERN == 8);

            int row, col;
            if (WARPS_4x1x1)
            {
                row = (tidx & 0x10) / 2 + (tidx & 0x07);
                col = (tidx & 0x60) / 16 + (tidx & 0x08) / 8;
            }
            else if (WARPS_4x1x2)
            {
                // Same as above, with second warp group writing next 16 rows.
                row = (tidx & 0x80) / 8 + (tidx & 0x10) / 2 + (tidx & 0x07);
                col = (tidx & 0x60) / 16 + (tidx & 0x08) / 8;
            }
            else
            {
                assert(false);
            }
            col ^= (row & 0x07);
            int dst = smem_tmp + row * BYTES_PER_ROW + col * BYTES_PER_LDS;

#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
#pragma unroll
                for (int ki = 0; ki < K; ki++)
                {
                    fmha::stsm(dst + ki * BYTES_PER_ROW * D, regs[ni][ki]);
                }
                if (WARPS_4x1x1 && N == 4)
                { // D=64, 1 warp in N.
                    dst += 16 * BYTES_PER_ROW;
                }
                else if (WARPS_4x1x2 && N == 2)
                { // D=64, 2 warps in N.
                    dst += 32 * BYTES_PER_ROW;
                }
                else
                {
                    assert(false);
                }
            }
        }

        Store_delegate delegate;
        Transposer transposer;
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF,
    // The description of the tile computed by this CTA.
    typename Cta_tile, int BUFFERS_PER_TILE,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // USe TMA or not,
    bool USE_TMA>
struct Smem_tile_v<fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    BUFFERS_PER_TILE, desc_mode, USE_TMA>
    : public Smem_tile_v_gmma<fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile, BUFFERS_PER_TILE, desc_mode, USE_TMA>::Base
{

    using Traits = fmha::Hopper_qgmma_e4m3_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = typename fmha::Smem_tile_v_gmma<Traits, Cta_tile, BUFFERS_PER_TILE, desc_mode, USE_TMA>::Base;

    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF,
    // The description of the tile computed by this CTA.
    typename Cta_tile, int BUFFERS_PER_TILE,
    // GMMA descriptor mode
    fmha::Gmma_descriptor_mode desc_mode,
    // USe TMA or not,
    bool USE_TMA>
struct Smem_tile_v<fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Cta_tile,
    BUFFERS_PER_TILE, desc_mode, USE_TMA>
    : public Smem_tile_v_gmma<fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>,
          Cta_tile, BUFFERS_PER_TILE, desc_mode, USE_TMA>::Base
{

    using Traits = fmha::Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>;

    using Base = typename fmha::Smem_tile_v_gmma<Traits, Cta_tile, BUFFERS_PER_TILE, desc_mode, USE_TMA>::Base;

    inline __device__ Smem_tile_v(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

} // namespace fmha
