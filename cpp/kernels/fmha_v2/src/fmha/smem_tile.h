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
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // The number of rows in the 2D shared memory buffer.
    int M_,
    // The number of cols.
    int N_,
    // The size in bits of each element.
    int BITS_PER_ELEMENT_,
    // The number of bytes per STS.
    int BYTES_PER_STS_ = 16,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1,
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_ = 0,
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS.
    int ROWS_PER_XOR_PATTERN_ = 8,
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS.
    int COLS_PER_XOR_PATTERN_ = 1,
    // Use or not predicates
    bool USE_PREDICATES_ = true,
    // Use TMA or not,
    bool USE_TMA_ = false,
    // The leading dim elements in shared memory
    int LEAD_DIM_ELEMENTS_ = N_>
struct Smem_tile_without_skews
{

    // The type of this tile
    using Smem_tile_ = Smem_tile_without_skews<Cta_tile, M_, N_, BITS_PER_ELEMENT_, BYTES_PER_STS_, BUFFERS_PER_TILE_,
        ENABLE_LDS_FAST_PATH_, ROWS_PER_XOR_PATTERN_, COLS_PER_XOR_PATTERN_, USE_PREDICATES_>;

    static constexpr bool USE_TMA = USE_TMA_;

    // The size in bits of each element.
    enum
    {
        BITS_PER_ELEMENT = BITS_PER_ELEMENT_
    };

    // The size in bytes of a single STS.
    enum
    {
        BYTES_PER_STS = BYTES_PER_STS_
    };

    // The number of elements per STS.
    enum
    {
        ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT
    };

    // To support arbitrary N, we pad some values to a power-of-2.
    enum
    {
        N_WITH_PADDING = Next_power_of_two<LEAD_DIM_ELEMENTS_>::VALUE
    };

    // The number of bytes per row without packing of rows.
    enum
    {
        BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8
    };

    // The number of bytes per row -- we want at least 128B per row.
    enum
    {
        BYTES_PER_ROW = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE
    };

    // The number of rows in shared memory (two rows may be packed into a single one).
    enum
    {
        ROWS = M_ * N_ / LEAD_DIM_ELEMENTS_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW
    };

    // The number of threads per row.
    enum
    {
        THREADS_PER_ROW_UNBOUNDED = BYTES_PER_ROW / BYTES_PER_STS
    };

    // The number of threads per row.
    enum
    {
        THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_UNBOUNDED>::VALUE
    };

    // The number of STS per row.
    enum
    {
        STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS
    };

    // It must be at least one.
    static_assert(STS_PER_ROW >= 1, "");

    // The number of rows written with a single STS.
    enum
    {
        ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // Make sure we write to at least one row per STS. Thanks Dr. Obvious ;)
    static_assert(ROWS_PER_STS >= 1, "");

    // The number of STS needed to store all rows.
    enum
    {
        STS_PER_COL = Div_up<ROWS, ROWS_PER_STS>::VALUE
    };

    // The number of STS in total.
    enum
    {
        STS = STS_PER_COL * STS_PER_ROW
    };

    // The size of one buffer in bytes in shared memory.
    enum
    {
        BYTES_PER_BUFFER = STS * BYTES_PER_STS * Cta_tile::THREADS_PER_CTA
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

    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS.
    enum
    {
        ROWS_PER_XOR_PATTERN = ROWS_PER_XOR_PATTERN_
    };

    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS.
    enum
    {
        COLS_PER_XOR_PATTERN = COLS_PER_XOR_PATTERN_ * 16 / BYTES_PER_STS
    };

    // Use or not predicates
    enum
    {
        USE_PREDICATES = USE_PREDICATES_
    };

    // The bytes of one shmem row
    enum
    {
        BYTES_PER_SHMEM_ROW = 128
    };

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type;

    // Ctor.
    inline __device__ Smem_tile_without_skews(void* smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem))
    {

        // The row written by a thread. See doc/mma_smem_layout.xlsx.
        int smem_write_row = tidx / THREADS_PER_ROW;

        // The XOR pattern.
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN * COLS_PER_XOR_PATTERN;
        // Compute the column and apply the XOR pattern.
        int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;

        // The offset.
        this->smem_write_offset_ = smem_write_row * BYTES_PER_ROW + smem_write_col * BYTES_PER_STS;

        // That code is expected to trigger the utilization of the URF by the compiler.
        this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template <int N, int K = 1>
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N])
    {

#pragma unroll
        for (int ii = 0; ii < N; ++ii)
        {
            // Decompose the STS into row/col.
            int row = ii % STS_PER_COL;
            int col = ii / STS_PER_COL;

            // Compute the immediate.
            int imm = row;

            // Assemble the offset.
            int offset = smem_write_offset_ + imm * ROWS_PER_STS * BYTES_PER_ROW;

            // Take the column into account.
            if (STS_PER_ROW > 1)
            {
                offset += col * THREADS_PER_ROW * BYTES_PER_STS;
            }

            // Apply the XOR pattern if needed.
            if (ROWS_PER_STS < ROWS_PER_XOR_PATTERN)
            {
                int const m = row * ROWS_PER_STS % ROWS_PER_XOR_PATTERN;
                offset ^= m * COLS_PER_XOR_PATTERN * BYTES_PER_STS;
            }

// Assemble the final pointer :)
#pragma unroll
            for (int k = 0; k < K; k++)
            {
                ptrs[ii * K + k] = smem_ + offset + k * (BYTES_PER_STS / K) + smem_write_buffer_;
            }
        }
    }

    inline __device__ void debug_reset()
    {
        for (int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER)
        {
            for (int row = 0; row < ROWS; ++row)
            {
                for (int col = 0; col < BYTES_PER_ROW; col += 4)
                {
                    if (threadIdx.x == 0)
                    {
                        uint32_t val = 0x0;
                        sts(val, smem_ + row * BYTES_PER_ROW + col + buffer);
                    }
                }
            }
        }
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const
    {
        for (int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER)
        {
            for (int row = 0; row < ROWS; ++row)
            {
                for (int col = 0; col < BYTES_PER_ROW; col += 4)
                {
                    if (threadIdx.x == 0)
                    {
                        uint32_t val;
                        lds(val, smem_ + row * BYTES_PER_ROW + col + buffer);
                        printf(
                            "block=(x=%2d, y=%2d, z=%2d) (smem_=0x%08x, buffer=%2d, row=%2d, "
                            "byte=%4d)=0x%08x\n",
                            blockIdx.x, blockIdx.y, blockIdx.z, smem_, buffer, row, col, val);
                    }
                }
            }
        }
    }

    // Move the read offset to next buffer.
    inline __device__ void move_to_next_read_buffer()
    {
        if (BUFFERS_PER_TILE > 1 && smem_read_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY)
        {
            this->smem_read_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        }
        else if (BUFFERS_PER_TILE > 1)
        {
            this->smem_read_buffer_ += BYTES_PER_BUFFER;
        }
    }

    // Move the read offset to next buffer. TODO: Remove this member function!!!
    inline __device__ void move_next_read_buffer()
    {
        this->move_to_next_read_buffer();
    }

    // Move the read offset to next N buffer (circular-buffer).
    inline __device__ void move_to_next_read_buffer(int N)
    {
        if (BUFFERS_PER_TILE > 1)
        {
            this->smem_read_buffer_ += N * BYTES_PER_BUFFER;
            this->smem_read_buffer_ -= smem_read_buffer_ >= BYTES_PER_TILE ? BYTES_PER_TILE : 0;
        }
    }

    // Move the read offset to next N buffer (circular-buffer). TODO: Remove this member function!!!
    inline __device__ void move_next_read_buffer(int N)
    {
        this->move_to_next_read_buffer(N);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_to_next_write_buffer()
    {
        if (BUFFERS_PER_TILE > 1 && smem_write_buffer_ >= BYTES_PER_TILE_INC_BOUNDARY)
        {
            this->smem_write_buffer_ -= BYTES_PER_TILE_INC_BOUNDARY;
        }
        else if (BUFFERS_PER_TILE > 1)
        {
            this->smem_write_buffer_ += BYTES_PER_BUFFER;
        }
    }

    // Move the write offset to next buffer. TODO: Remove that member function!
    inline __device__ void move_next_write_buffer()
    {
        this->move_to_next_write_buffer();
    }

    // Move the read offset.
    inline __device__ void move_read_offset(int delta)
    {
        this->smem_read_offset_ += delta;
    }

    // Move the write offset.
    inline __device__ void move_write_offset(int delta)
    {
        this->smem_write_offset_ += delta;
    }

    // Store to the tile in shared memory.
    template <int N>
    inline __device__ void store(Store_type const (&data)[N])
    {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data);
    }

    // Store to the tile in shared memory.
    template <int N, int M>
    inline __device__ void store(Store_type const (&data)[N], uint32_t (&preds)[M])
    {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        sts(smem_ptrs, data, preds);
    }

    // Store to the tile in shared memory.
    template <int N>
    inline __device__ void store(Store_type const (&data)[N], uint32_t preds)
    {
        this->store(data, preds);
    }

    // Store to the tile in shared memory. TODO: Remove last template arguments.
    template <int N, int M>
    inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t (&preds)[M])
    {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers<N>(smem_ptrs);
        ldgsts<N, M>(smem_ptrs, gmem_ptrs, preds);
    }

    // Store to the tile in shared memory.
    template <int N>
    inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t preds, uint64_t = 0)
    {
        uint32_t tmp[1] = {preds};
        this->store(gmem_ptrs, tmp);
    }

    // Store to the tile in shared memory.
    template <int N>
    inline __device__ void store(void const* (&gmem_ptrs)[N], uint32_t preds)
    {
        uint32_t tmp[1] = {preds};
        this->store(gmem_ptrs, tmp);
    }

    inline __device__ void add_smem_barrier_base(uint64_t*) {}

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
// Use TMA
template <
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // The number of rows in the 2D shared memory buffer.
    int M_,
    // The number of cols.
    int N_,
    // The size in bits of each element.
    int BITS_PER_ELEMENT_,
    // The number of bytes per STS. Not relevant for TMA
    int BYTES_PER_STS_,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_,
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_,
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS.
    int ROWS_PER_XOR_PATTERN_,
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS.
    int COLS_PER_XOR_PATTERN_,
    // Use or not predicates
    bool USE_PREDICATES_,
    // The leading dim elements in shared memory
    int LEAD_DIM_ELEMENTS_>
struct Smem_tile_without_skews<Cta_tile, M_, N_, BITS_PER_ELEMENT_, BYTES_PER_STS_, BUFFERS_PER_TILE_,
    ENABLE_LDS_FAST_PATH_, ROWS_PER_XOR_PATTERN_, COLS_PER_XOR_PATTERN_, USE_PREDICATES_, true, LEAD_DIM_ELEMENTS_>
    : public Smem_tile_without_skews<Cta_tile, M_, N_, BITS_PER_ELEMENT_, BYTES_PER_STS_, BUFFERS_PER_TILE_,
          ENABLE_LDS_FAST_PATH_, ROWS_PER_XOR_PATTERN_, COLS_PER_XOR_PATTERN_, USE_PREDICATES_, false,
          LEAD_DIM_ELEMENTS_>
{
    // Base struct
    using Base = Smem_tile_without_skews<Cta_tile, M_, N_, BITS_PER_ELEMENT_, BYTES_PER_STS_, BUFFERS_PER_TILE_,
        ENABLE_LDS_FAST_PATH_, ROWS_PER_XOR_PATTERN_, COLS_PER_XOR_PATTERN_, USE_PREDICATES_, false,
        LEAD_DIM_ELEMENTS_>;
    static constexpr bool USE_TMA = true;

    // Tile size overrides. STS per thread not relevant for TMA
    static constexpr int BYTES_PER_BUFFER = M_ * N_ * Base::BITS_PER_ELEMENT / 8;
    static constexpr int BYTES_PER_TILE = BYTES_PER_BUFFER * Base::BUFFERS_PER_TILE;
    static constexpr int BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER;
    // The number of bytes per barrier
    static constexpr int BYTES_PER_BARRIER = 8;

    // Ctor
    inline __device__ Smem_tile_without_skews(void* smem, int tidx)
        : Base(smem, tidx)
    {
        this->smem_write_offset_ = __nvvm_get_smem_pointer(smem);
        this->smem_barrier_offset_ = 0;
        this->elect_one_ = elect_one_sync();
    }

    inline __device__ void add_smem_barrier_base(uint64_t* smem_barrier)
    {
        this->smem_barrier_ = smem_barrier;
        this->smem_barrier_offset_ = __nvvm_get_smem_pointer(this->smem_barrier_);
    }

    /**
     * \brief load tensor blocks from global memory and stores to shared memory using tma instructions
     *
     * \param p_desc pointer to tma descriptor masked as const void* pointer
     * \param smem_offset shared memory offset in bytes relative to smem_write_buffer_
     * \param coord0 tensor access coordinate in dimension 1, used by tma load
     * \param coord1 tensor access coordinate in dimension 2, used by tma load
     * \param coord2 tensor access coordinate in dimension 3, used by tma load
     * \param coord3 tensor access coordinate in dimension 4, used by tma load
     * \param coord4 tensor access coordinate in dimension 5, used by tma load
     * \param filter_offsets encodes multicast cta id and filter offsets
     */
    template <uint32_t DIM, cudaTmaDescType DESC_TYPE, unsigned COPY_BYTES, bool USE_TMA_MULTICAST = false>
    inline __device__ void store(void const* p_desc, unsigned const& smem_offset, int32_t coord0, int32_t coord1,
        int32_t coord2, int32_t coord3, int32_t coord4, uint16_t filter_offsets, uint16_t mcast_cta_mask,
        uint64_t mem_desc)
    {
        uint32_t smem = this->smem_write_offset_ + smem_offset;
        fmha::utmaldg<DIM, DESC_TYPE, USE_TMA_MULTICAST>(reinterpret_cast<cudaTmaDesc const*>(p_desc), smem,
            unsigned(this->smem_barrier_offset_), coord0, coord1, coord2, coord3, coord4, filter_offsets,
            mcast_cta_mask, mem_desc, this->elect_one_);
    }

    // Same function as above but for runtime cga dimension
    template <uint32_t DIM, cudaTmaDescType DESC_TYPE, unsigned COPY_BYTES>
    inline __device__ void store(void const* p_desc, unsigned const& smem_offset, int32_t coord0, int32_t coord1,
        int32_t coord2, int32_t coord3, int32_t coord4, uint16_t filter_offsets, uint16_t mcast_cta_mask,
        uint64_t mem_desc, bool mcast_enabled)
    {
        uint32_t smem = this->smem_write_offset_ + smem_offset;
        fmha::utmaldg<DIM, DESC_TYPE>(reinterpret_cast<cudaTmaDesc const*>(p_desc), smem,
            unsigned(this->smem_barrier_offset_), coord0, coord1, coord2, coord3, coord4, filter_offsets,
            mcast_cta_mask, mcast_enabled, mem_desc, this->elect_one_);
    }

    // Move the write offset to next buffer.
    inline __device__ void move_next_write_buffer()
    {
        if (Base::BUFFERS_PER_TILE > 1)
        {
            this->smem_write_offset_ += (this->smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY)
                ? -BYTES_PER_TILE_INC_BOUNDARY
                : BYTES_PER_BUFFER;
            this->smem_barrier_offset_ += (this->smem_barrier_offset_ >= Base::BUFFERS_PER_TILE * BYTES_PER_BARRIER)
                ? -Base::BUFFERS_PER_TILE * BYTES_PER_BARRIER
                : BYTES_PER_BARRIER;
        }
    }

    inline __device__ void move_next_write_buffer(int buffer_id)
    {
        if (Base::BUFFERS_PER_TILE > 1)
        {
            this->smem_write_offset_ = this->smem_ + buffer_id * BYTES_PER_BUFFER;
        }
        this->smem_barrier_offset_ = __nvvm_get_smem_pointer(this->smem_barrier_ + buffer_id);
    }

    // Move the read offset to next buffer.
    // do nothing, as it is controlled by gmma desc
    inline __device__ void move_next_read_buffer() {}

    uint64_t* smem_barrier_;
    uint32_t smem_barrier_offset_;
    // elect one thread to issue utmaldg
    uint32_t elect_one_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true>
struct Smem_tile_a
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_volta_a
{
    // The size in bits.
    enum
    {
        N_IN_BITS = N * Traits::BITS_PER_ELEMENT_A
    };

    // The number of rows.
    enum
    {
        VALUE = N_IN_BITS <= 256 ? 1 : (N_IN_BITS <= 512 ? 2 : 4)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_K, int MMAS_K_WITH_PADDING>
struct Compute_reset_mask
{
    // The potential mask.
    enum
    {
        HALF = MMAS_K_WITH_PADDING / 2
    };

    // The remainder.
    enum
    {
        MOD = MMAS_K % HALF
    };

    // The final value.
    enum
    {
        VALUE = (MMAS_K == MOD ? 0 : HALF) | Compute_reset_mask<MOD, HALF>::VALUE
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_K_WITH_PADDING>
struct Compute_reset_mask<0, MMAS_K_WITH_PADDING>
{
    enum
    {
        VALUE = 0
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMAS_K>
struct Compute_reset_mask<MMAS_K, MMAS_K>
{
    enum
    {
        VALUE = MMAS_K - 1
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_volta_a<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_volta_row_a : public Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K, 16, BYTES_PER_STS,
                                   BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K, 16, BYTES_PER_STS, BUFFERS_PER_TILE, 0,
        ROWS_PER_XOR_PATTERN_, 1>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = typename Traits::template Mma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_volta_row_a(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;
        if (Base::N_WITH_PADDING >= 64)
        {
            smem_read_row
                = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 1 + (tidx & 0x10) / 2 + (tidx & 0x07);
            smem_read_col = (tidx & 0x03);
        }
        else if (Base::N_WITH_PADDING == 32)
        {
            smem_read_row
                = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 2 + (tidx & 0x10) / 4 + (tidx & 0x06) / 2;
            smem_read_col = (tidx & 0x02) / 2 + (tidx & 0x01) * 4;
        }
        else
        {
            assert(false);
        }

        // For WARPS_K > 1, we do not support Base::N_WITH_PADDING < 64 for the moment.
        static_assert(WARPS_K <= 2 && (WARPS_K == 1 || Base::N_WITH_PADDING >= 64), "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if (WARPS_K == 2)
        {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile_with_padding::MMAS_K;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // Move the offset to the next position. See doc/xmma_smem_layout.xlsx.
        this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki)
    {
#pragma unroll
        for (int mi = 0; mi < Mma_tile::MMAS_M; ++mi)
        {
            // Jump over as many rows as needed.
            int offset = mi * Mma_tile::M_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // TODO: Could we fuse smem_read_buffer and smem_read_offset?
            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // Move the offset to the next position. See doc/xmma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15)
        {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7)
        {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3)
        {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1)
        {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset()
    {
        // The number of MMAs in the K dimension.
        enum
        {
            MMAS_K = Mma_tile::MMAS_K
        };

        // The number of MMAs in the K dimension when we include padding.
        enum
        {
            MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K
        };

        // Assemble the mask.
        enum
        {
            MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE
        };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Volta_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_volta_row_a<Volta_hmma_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = fmha::Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_turing_a
{
    // The size in bits.
    enum
    {
        N_IN_BITS = N * Traits::BITS_PER_ELEMENT_A
    };

    // The number of rows.
    enum
    {
        VALUE = N_IN_BITS <= 128 ? 1 : (N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8))
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_turing_a<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_turing_row_a
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K, Traits::BITS_PER_ELEMENT_A, BYTES_PER_STS,
          BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K, Traits::BITS_PER_ELEMENT_A, BYTES_PER_STS,
        BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = typename Traits::template Mma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_turing_row_a(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/mma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::ROWS_PER_XOR_PATTERN == 8 || Base::ROWS_PER_XOR_PATTERN == 4
                || Base::ROWS_PER_XOR_PATTERN == 2 || Base::ROWS_PER_XOR_PATTERN == 1,
            "");

        if (Base::ROWS_PER_XOR_PATTERN == 8)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 1 + (tidx & 0x0f);
            smem_read_col = (tidx & 0x07);
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 4)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 2 + (tidx & 0x0e) / 2;
            smem_read_col = (tidx & 0x06) / 2 + (tidx & 0x01) * 4;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 2)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 4 + (tidx & 0x0c) / 4;
            smem_read_col = (tidx & 0x04) / 4 + (tidx & 0x03) * 2;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 1)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 8 + (tidx & 0x1f) / 8;
            smem_read_col = (tidx & 0x07);
        }

        static_assert(WARPS_K <= 2, "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if (WARPS_K == 2)
        {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile_with_padding::MMAS_K;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // Move the offset to the next position. See doc/mma_smem_layout.xlsx.
        this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki)
    {
#pragma unroll
        for (int mi = 0; mi < Mma_tile::MMAS_M; ++mi)
        {
            int offset = mi * Mma_tile::M_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;
            uint2 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
        }

        // Move the offset to the next position. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15)
        {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7)
        {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3)
        {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1)
        {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset()
    {
        // The number of MMAs in the K dimension.
        enum
        {
            MMAS_K = Mma_tile::MMAS_K
        };

        // The number of MMAs in the K dimension when we include padding.
        enum
        {
            MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K
        };

        // Assemble the mask.
        enum
        {
            MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE
        };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Turing_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_row_a<Turing_hmma_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Turing_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_row_a<Turing_hmma_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Turing_imma_int8_int32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_row_a<Turing_imma_int8_int32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Turing_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_ampere_a
{
    // The size in bits.
    enum
    {
        N_IN_BITS = N * Traits::BITS_PER_ELEMENT_A
    };

    // The number of rows.
    enum
    {
        VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_ampere_row_a : public Rows_per_xor_pattern_ampere_a<Traits, N>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_row_a<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_ampere_row_a
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K, Traits::BITS_PER_ELEMENT_A, BYTES_PER_STS,
          BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>
{
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::M, Cta_tile::K, Traits::BITS_PER_ELEMENT_A, BYTES_PER_STS,
        BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>;
    // The fragment.
    using Fragment = Fragment_a<Traits, Row>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = typename Traits::template Mma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_ampere_row_a(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/mma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_M = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::M;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_M = 1 * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(
            Base::ROWS_PER_XOR_PATTERN == 8 || Base::ROWS_PER_XOR_PATTERN == 4 || Base::ROWS_PER_XOR_PATTERN == 2, "");

        if (Base::ROWS_PER_XOR_PATTERN == 8)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 1 + (tidx & 0x0f);
            smem_read_col = (tidx & 0x07);
            smem_read_col ^= (tidx & 0x10) / 16;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 4)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 2 + (tidx & 0x0e) / 2;
            smem_read_col = (tidx & 0x06) / 2 + (tidx & 0x01) * 4;
            smem_read_col ^= (tidx & 0x10) / 16;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 2)
        {
            smem_read_row = (tidx & WARP_MASK_M) / WARP_DIV_M * Mma_tile::M_PER_MMA / 4 + (tidx & 0x0c) / 4;
            smem_read_col = (tidx & 0x04) / 4 + (tidx & 0x03) * 2;
            smem_read_col ^= (tidx & 0x10) / 16;
        }

        static_assert(WARPS_K <= 2, "");
        static_assert(WARPS_K != 2 || Base::ROWS_PER_XOR_PATTERN != 2, "");

        // We "swap" the block for the second warp working on the same outputs in-CTA split-K.
        if (WARPS_K == 2)
        {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile_with_padding::MMAS_K * 2;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // Undo the pointer increment for the next ni.
        // Should match the load function below for ki = 0.
        if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki)
    {
        if (ki < Mma_tile::VALID_MMAS_K)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile::MMAS_M; ++mi)
            {
                // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
                int offset = mi * Mma_tile::M_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

                // Load using LDSM.M88.4.
                uint4 tmp;
                ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

                // Store the value into the fragment.
                a[mi].reg(0) = tmp.x;
                a[mi].reg(1) = tmp.y;
                a[mi].reg(2) = tmp.z;
                a[mi].reg(3) = tmp.w;
            }
        }

        // Move the offset to the next position. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15)
        {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7)
        {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3)
        {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1)
        {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset()
    {
        // The number of MMAs in the K dimension.
        enum
        {
            MMAS_K = Mma_tile::MMAS_K
        };

        // The number of MMAs in the K dimension when we include padding.
        enum
        {
            MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K
        };

        // Assemble the mask.
        enum
        {
            MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE
        };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Ampere_hmma_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{
    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Ampere_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{
    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Ampere_hmma_bf16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_hmma_bf16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{
    // The traits class.
    using Traits = Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Ampere_imma_int8_int32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ampere_imma_int8_int32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Ada_qmma_e4m3_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ada_qmma_e4m3_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_a<Ada_qmma_e4m3_fp16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_a<Ada_qmma_e4m3_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ada_qmma_e4m3_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_a<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true>
struct Smem_tile_b
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_volta_b
{
    // The size in bits.
    enum
    {
        N_IN_BITS = N * Traits::BITS_PER_ELEMENT_B
    };

    // The number of rows.
    enum
    {
        VALUE = N_IN_BITS <= 256 ? 1 : (N_IN_BITS <= 512 ? 2 : 4)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_volta_b<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_volta_col_b : public Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K, 16, BYTES_PER_STS,
                                   BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K, 16, BYTES_PER_STS, BUFFERS_PER_TILE, 0,
        ROWS_PER_XOR_PATTERN_, 1>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = typename Traits::template Mma_tile<Cta_tile_with_padding>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_volta_col_b(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        if (Base::N_WITH_PADDING >= 64)
        {
            smem_read_row
                = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 1 + (tidx & 0x18) / 2 + (tidx & 0x03);
            smem_read_col = (tidx & 0x03);
        }
        else if (Base::N_WITH_PADDING == 32)
        {
            smem_read_row
                = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 2 + (tidx & 0x18) / 4 + (tidx & 0x02) / 2;
            smem_read_col = (tidx & 0x02) / 2 + (tidx & 0x01) * 4;
        }
        else
        {
            assert(false);
        }

        // For WARPS_K > 1, we do not support Base::N_WITH_PADDING < 64 for the moment.
        static_assert(WARPS_K <= 2 && (WARPS_K == 1 || Base::N_WITH_PADDING >= 64), "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if (WARPS_K == 2)
        {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile_with_padding::MMAS_K;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // Move the offset to the next position. See doc/xmma_smem_layout.xlsx.
        this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {

            // Jump over as many rows as needed.
            int offset = ni * Mma_tile::N_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // TODO: Can we fuse read_offset and read_buffer?
            uint4 tmp;
            lds(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
        }

        // Move the offset to the next position. See doc/xmma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15)
        {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7)
        {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3)
        {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1)
        {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset()
    {
        // The number of MMAs in the K dimension.
        enum
        {
            MMAS_K = Mma_tile::MMAS_K
        };

        // The number of MMAs in the K dimension when we include padding.
        enum
        {
            MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K
        };

        // Assemble the mask.
        enum
        {
            MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE
        };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Volta_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_volta_col_b<Volta_hmma_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = fmha::Volta_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_volta_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_turing_b
{
    // The size in bits.
    enum
    {
        N_IN_BITS = N * Traits::BITS_PER_ELEMENT_B
    };

    // The number of rows.
    enum
    {
        VALUE = N_IN_BITS <= 128 ? 1 : (N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8))
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_turing_b<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_turing_col_b
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS,
          BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS,
        BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = typename Traits::template Mma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // Ctor.
    inline __device__ Smem_tile_turing_col_b(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/mma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(Base::ROWS_PER_XOR_PATTERN == 8 || Base::ROWS_PER_XOR_PATTERN == 4
                || Base::ROWS_PER_XOR_PATTERN == 2 || Base::ROWS_PER_XOR_PATTERN == 1,
            "");

        if (Base::ROWS_PER_XOR_PATTERN == 8)
        {
            // For group fprop. B is divided into 2 halves along N dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 1 + (tidx & 0x0f);
            smem_read_col = (tidx & 0x07);
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 4)
        {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 2 + (tidx & 0x0e) / 2;
            smem_read_col = (tidx & 0x06) / 2 + (tidx & 0x01) * 4;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 2)
        {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 4 + (tidx & 0x0c) / 4;
            smem_read_col = (tidx & 0x04) / 4 + (tidx & 0x03) * 2;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 1)
        {
            smem_read_row = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 8 + (tidx & 0x1f) / 8;
            smem_read_col = (tidx & 0x07);
        }

        static_assert(WARPS_K <= 2, "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if (WARPS_K == 2)
        {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile_with_padding::MMAS_K;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.-
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // Move the offset to the next position. See doc/mma_smem_layout.xlsx.
        this->smem_read_offset_ ^= ((ki % 2 == 0) ? 1 : 3) * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki)
    {
#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            int offset = ni * Mma_tile::N_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;
            uint2 tmp;
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
        }
        // Move the offset to the next position. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15)
        {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7)
        {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3)
        {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1)
        {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset()
    {
        // The number of MMAs in the K dimension.
        enum
        {
            MMAS_K = Mma_tile::MMAS_K
        };

        // The number of MMAs in the K dimension when we include padding.
        enum
        {
            MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K
        };

        // Assemble the mask.
        enum
        {
            MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE
        };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Turing_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_hmma_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Turing_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Turing_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_hmma_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Turing_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Turing_imma_int8_int32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_turing_col_b<Turing_imma_int8_int32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Turing_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_turing_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_ampere_b
{
    // The size in bits.
    enum
    {
        N_IN_BITS = N * Traits::BITS_PER_ELEMENT_B
    };

    // The number of rows.
    enum
    {
        VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8)
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_ampere_col_b : public Rows_per_xor_pattern_ampere_b<Traits, N>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_col_b<Traits, Cta_tile::K>::VALUE>
struct Smem_tile_ampere_col_b
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS,
          BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>
{
    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::N, Cta_tile::K, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS,
        BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, 1>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Traits, Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = typename Traits::template Mma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = 16
    };

    // The number of STS per thread
    enum
    {
        STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA
    };

    // The number of STS per thread must be at least 1.
    enum
    {
        STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE
    };

    // Ctor.
    inline __device__ Smem_tile_ampere_col_b(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/mma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row and column read by the thread.
        int smem_read_row, smem_read_col;

        static_assert(
            Base::ROWS_PER_XOR_PATTERN == 8 || Base::ROWS_PER_XOR_PATTERN == 4 || Base::ROWS_PER_XOR_PATTERN == 2, "");

        if (Base::ROWS_PER_XOR_PATTERN == 8)
        {
            // For group fprop. B is divided into 2 halves along N dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            smem_read_row
                = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 1 + (tidx & 0x07) + (tidx & 0x10) / 2;
            smem_read_col = (tidx & 0x07);
            smem_read_col ^= (tidx & 0x08) / 8;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 4)
        {
            smem_read_row
                = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 2 + (tidx & 0x06) / 2 + (tidx & 0x10) / 4;
            smem_read_col = (tidx & 0x06) / 2 + (tidx & 0x01) * 4;
            smem_read_col ^= (tidx & 0x08) / 8;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 2)
        {
            smem_read_row
                = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA / 4 + (tidx & 0x04) / 4 + (tidx & 0x10) / 8;
            smem_read_col = (tidx & 0x04) / 4 + (tidx & 0x03) * 2;
            smem_read_col ^= (tidx & 0x08) / 8;
        }

        static_assert(WARPS_K <= 2, "");
        static_assert(WARPS_K != 2 || Base::ROWS_PER_XOR_PATTERN != 2, "");

        // We "swap" the block for the second warp working on the in-CTA split-K.
        if (WARPS_K == 2)
        {
            smem_read_col ^= (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile_with_padding::MMAS_K * 2;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // Undo the pointer increment for the next ni.
        // Should match the load function below for ki = 0.
        if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki)
    {
        if (ki < Mma_tile::VALID_MMAS_K)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
            {
                // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
                int offset = ni * Mma_tile::N_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

                // Load using LDSM.M88.4.
                uint4 tmp;
                ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);

                // Store the value into the fragment.
                b[ni].reg(0) = tmp.x;
                b[ni].reg(1) = tmp.y;
                b[ni].reg(2) = tmp.z;
                b[ni].reg(3) = tmp.w;
            }
        }

        // Move the offset to the next position. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if (Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15)
        {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 16 && ki % 8 == 7)
        {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 8 && ki % 4 == 3)
        {
            this->smem_read_offset_ ^= 7 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 4 && ki % 2 == 1)
        {
            this->smem_read_offset_ ^= 3 * BYTES_PER_LDS * 2;
        }
        else if (Mma_tile_with_padding::MMAS_K >= 2)
        {
            this->smem_read_offset_ ^= 1 * BYTES_PER_LDS * 2;
        }
    }

    // Reset the read offset.
    inline __device__ void reset_read_offset()
    {
        // The number of MMAs in the K dimension.
        enum
        {
            MMAS_K = Mma_tile::MMAS_K
        };

        // The number of MMAs in the K dimension when we include padding.
        enum
        {
            MMAS_K_WITH_PADDING = Mma_tile_with_padding::MMAS_K
        };

        // Assemble the mask.
        enum
        {
            MASK = Compute_reset_mask<MMAS_K, MMAS_K_WITH_PADDING>::VALUE
        };

        // Reset the read offset.
        this->smem_read_offset_ ^= MASK * BYTES_PER_LDS * 2;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ampere_hmma_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_hmma_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ampere_hmma_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ampere_hmma_bf16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_hmma_bf16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ampere_imma_int8_int32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ampere_imma_int8_int32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_imma_int8_int32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ada_qmma_e4m3_fp32_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ada_qmma_e4m3_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ada_qmma_e4m3_fp16_traits, Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_col_b<Ada_qmma_e4m3_fp16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ada_qmma_e4m3_fp16_traits;
    // The base class.
    using Base = Smem_tile_ampere_col_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, int N>
struct Rows_per_xor_pattern_ampere_row_b : public Rows_per_xor_pattern_ampere_b<Traits, N>
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_ampere_row_b<Traits, Cta_tile::N>::VALUE,
    // How many cols to use for the XOR pattern to avoid bank conflicts?
    int COLS_PER_XOR_PATTERN_ = 1>
struct Smem_tile_ampere_row_b
    : public Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS,
          BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, COLS_PER_XOR_PATTERN_>
{

    // The MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, Traits::BITS_PER_ELEMENT_B, BYTES_PER_STS,
        BUFFERS_PER_TILE, 0, ROWS_PER_XOR_PATTERN_, COLS_PER_XOR_PATTERN_>;
    // The fragment.
    using Fragment = Fragment_b<Traits, Row>;

    // Can we use LDSM? No if the data type is 32-bit large.
    enum
    {
        USE_LDSMT = Traits::BITS_PER_ELEMENT_B == 16
    };

    // The size of a single LDS in bytes.
    enum
    {
        BYTES_PER_LDS = USE_LDSMT ? 16 : 4
    };

    // The number of elements per LDS.
    enum
    {
        ELEMENTS_PER_LDS = BYTES_PER_LDS * 8 / Traits::BITS_PER_ELEMENT_B
    };

    // The number of STS per thread
    enum
    {
        STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA
    };

    // The number of STS per thread must be at least 1.
    enum
    {
        STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE
    };

    // Ctor.
    inline __device__ Smem_tile_ampere_row_b(void* smem, int tidx)
        : Base(smem, tidx)
    {

        // For documentation on the layout, see doc/xmma_smem_layout.xlsx.

        // The number of warps.
        int const WARPS_M = Cta_tile::WARPS_M;
        int const WARPS_N = Cta_tile::WARPS_N;
        int const WARPS_K = Cta_tile::WARPS_K;

        // The masks to select the warps.
        int const WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        int const WARP_MASK_K = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::K;

        // The divisor for the warps.
        int const WARP_DIV_N = WARPS_M * 1 * Cta_tile::THREADS_PER_WARP;
        int const WARP_DIV_K = WARPS_M * WARPS_N * Cta_tile::THREADS_PER_WARP;

        // The row/col read by the thread.
        int smem_read_row, smem_read_col;

        static_assert((USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8) || Base::ROWS_PER_XOR_PATTERN == 4
                || Base::ROWS_PER_XOR_PATTERN == 2,
            "");

        if (USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 8)
        {
            // For group dgrad. B is divided into 2 halves along K dimension.
            // The fist warp takes the first half and the second warp takes the second half.
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile::MMAS_K * 16 + (tidx & 0x07) + (tidx & 0x08);
            smem_read_col = (tidx & 0x07);
        }
        else if (USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 4)
        {
            smem_read_row
                = (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile::MMAS_K * 8 + (tidx & 0x06) / 2 + (tidx & 0x08) / 2;
            smem_read_col = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        }
        else if (USE_LDSMT && Base::ROWS_PER_XOR_PATTERN == 2)
        {
            smem_read_row
                = (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile::MMAS_K * 4 + (tidx & 0x04) / 4 + (tidx & 0x08) / 4;
            smem_read_col = (tidx & 0x03) * 2 + (tidx & 0x04) / 4;
        }
        else if (Base::ROWS_PER_XOR_PATTERN == 4 && Base::COLS_PER_XOR_PATTERN == 2)
        {
            smem_read_row = (tidx & WARP_MASK_K) / WARP_DIV_K * Mma_tile::MMAS_K * 8 + (tidx & 0x03);
            smem_read_col = (tidx & 0x1c) / 4 + (tidx & 0x03) * 8;
        }

        // Each half-warp applies a different XOR pattern -- see the Excel document.
        if (USE_LDSMT)
        {
            smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 2 + (tidx & 0x10) / 16;
        }
        else
        {
            smem_read_col ^= (tidx & WARP_MASK_N) / WARP_DIV_N * 16;
        }

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row * Base::BYTES_PER_ROW + smem_read_col * BYTES_PER_LDS;

        // Fill zeroes for group conv
    }

    // Rewind smem_read_offset for last LDS phase in main loop.
    inline __device__ void reverse_smem_read_offset(int ki = 0)
    {
        // The size of each element in bits.
        int const BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B;
        // The size in bytes of the data needed to compute an MMA per CTA.
        int const BYTES_PER_MMA_PER_CTA = Mma_tile::N_PER_MMA_PER_CTA * BITS_PER_ELT / 8;

#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            // Undo the pointer increment for the next ni.
            // Should match the load function below for ki = 0.
            if (BYTES_PER_MMA_PER_CTA >= 128)
            {
                // Nothing to do!
            }
            else if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1)
            {
                this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
            }
            else if (BYTES_PER_MMA_PER_CTA == 64)
            {
                // Nothing to do!
            }
            else if (BYTES_PER_MMA_PER_CTA == 32 && Mma_tile::MMAS_N == 4)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            }
            else if (BYTES_PER_MMA_PER_CTA == 32 && Mma_tile::MMAS_N == 2)
            {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            }
        }

        // Reset smem_read_offset for odd MMAS_N > 1 (npo2 kernels)
        if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1 && Mma_tile::MMAS_N % 2 == 1)
        {
            this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
        }
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::VALID_MMAS_N], int ki)
    {
        // The size of each element in bits.
        int const BITS_PER_ELT = Traits::BITS_PER_ELEMENT_B;
        // The size in bytes of the data needed to compute an MMA per CTA.
        int const BYTES_PER_MMA_PER_CTA = Mma_tile::N_PER_MMA_PER_CTA * BITS_PER_ELT / 8;

#pragma unroll
        for (int ni = 0; ni < Mma_tile::MMAS_N; ++ni)
        {
            // Prepare the offset.
            int offset = ki * Base::ROWS_PER_XOR_PATTERN * 2 * Base::BYTES_PER_ROW;
            if (BYTES_PER_MMA_PER_CTA == 32)
            {
                offset += this->smem_read_offset_;
            }
            else if (BYTES_PER_MMA_PER_CTA == 64)
            {
                offset += this->smem_read_offset_ + (ni / 2) * BYTES_PER_MMA_PER_CTA * 2;
            }
            else
            {
                offset += this->smem_read_offset_ + (ni) *BYTES_PER_MMA_PER_CTA;
            }

            // Load the data using LDSM.MT88.2.
            uint32_t ptr = this->smem_ + this->smem_read_buffer_ + offset;

            if (ni < Mma_tile::VALID_MMAS_N)
            {
                uint4 tmp;
                if (USE_LDSMT)
                {
                    ldsmt(tmp, ptr);
                }
                else
                {
                    lds(tmp.x, (ptr) + 0 * Base::BYTES_PER_ROW);
                    lds(tmp.y, (ptr) + 4 * Base::BYTES_PER_ROW);
                    lds(tmp.z, (ptr ^ 32) + 0 * Base::BYTES_PER_ROW);
                    lds(tmp.w, (ptr ^ 32) + 4 * Base::BYTES_PER_ROW);
                }

                // Store those values in the fragment.
                b[ni].reg(0) = tmp.x;
                b[ni].reg(1) = tmp.y;
                b[ni].reg(2) = tmp.z;
                b[ni].reg(3) = tmp.w;
            }

            // static_assert(BYTES_PER_MMA_PER_CTA >= 128 ||
            //               BYTES_PER_MMA_PER_CTA ==  64 ||
            //               (BYTES_PER_MMA_PER_CTA == 32 &&
            //               (Mma_tile::MMAS_M == 4 ||
            //               Mma_tile::MMAS_M == 2 ||
            //               Mma_tile::MMAS_M == 1)), "");

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if (BYTES_PER_MMA_PER_CTA >= 128)
            {
                // Nothing to do!
            }
            else if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1)
            {
                this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
            }
            else if (BYTES_PER_MMA_PER_CTA == 64)
            {
                // Nothing to do!
            }
            else if (BYTES_PER_MMA_PER_CTA == 32)
            {
                if ((ni & 1) == 0)
                {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
                }
                else if (Mma_tile::MMAS_N >= 16 && (ni & 7) == 7)
                {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 30;
                }
                else if (Mma_tile::MMAS_N >= 8 && (ni & 3) == 3)
                {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 14;
                }
                else if (Mma_tile::MMAS_N >= 4 && (ni & 1) == 1)
                {
                    this->smem_read_offset_ ^= BYTES_PER_LDS * 6;
                }
            }
        }

        // Reset smem_read_offset for odd MMAS_N > 1 (npo2 kernels)
        if (BYTES_PER_MMA_PER_CTA == 64 && Mma_tile::MMAS_N > 1 && Mma_tile::MMAS_N % 2 == 1)
        {
            this->smem_read_offset_ ^= BYTES_PER_MMA_PER_CTA;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ampere_hmma_fp32_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_b<Ampere_hmma_fp32_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE>
struct Smem_tile_b<Ampere_hmma_bf16_traits, Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_ampere_row_b<Ampere_hmma_bf16_traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>
{

    // The traits class.
    using Traits = Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Smem_tile_ampere_row_b<Traits, Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void* smem, int tidx)
        : Base(smem, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
