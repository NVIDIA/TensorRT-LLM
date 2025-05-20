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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, int BYTES_PER_ELEMENT>
struct Gmem_tile_mma_sd
{
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    enum
    {
        BYTES_PER_STG = BYTES_PER_ELEMENT * 8
    };

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
        M_PER_MMA_PER_CTA = Mma_tile::M_PER_MMA_PER_CTA
    };

    enum
    {
        N_PER_MMA_PER_CTA = Mma_tile::N_PER_MMA_PER_CTA
    };

    enum
    {
        THREADS_PER_CTA = Cta_tile::THREADS_PER_CTA
    };

    enum
    {
        BYTES_PER_ROW = THREADS_PER_CTA * BYTES_PER_STG
    };

    // enum { SEQLEN = Cta_tile::N };

    // enum { BLOCK_STRIDE_BYTES = SEQLEN * SEQLEN * BYTES_PER_ELEMENT };
    enum
    {
        LOOP_STRIDE_BYTES = MMAS_M * MMAS_N * BYTES_PER_ROW
    };

    enum
    {
        D = Cta_tile::K
    };

    enum
    {
        STEPK = Cta_tile::N
    };

    enum
    {
        STEPQ = Cta_tile::M
    };

    using Type = typename fmha::Uint_from_size_in_bytes<BYTES_PER_STG>::Type;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_mma_sd(void* ptr, Params const& params, Block_info const& block_info, int tidx)
        : ptr_(static_cast<char*>(ptr))
    {
        int bidb = block_info.bidb;
        int bidh = block_info.bidh;
        actual_seqlen_ = block_info.actual_seqlen;
        assert(actual_seqlen_ % STEPK == 0);
        total_ksteps_ = actual_seqlen_ / STEPK;

        // The block index.
        size_t bidx = bidb * params.h + bidh;

        size_t block_stride_bytes = actual_seqlen_ * actual_seqlen_ * BYTES_PER_ELEMENT;

        // Set store location for each thread at the beginning of the loop
        ptr_ += bidx * block_stride_bytes + tidx * BYTES_PER_STG;
        init_ptr_ = ptr_;
    }

    inline __device__ void store(Type const& data, int mi, int ni)
    {
        size_t offset = (ni + mi * MMAS_N * total_ksteps_) * BYTES_PER_ROW;
        fmha::stg(ptr_ + offset, data);
    }

    inline __device__ void load(Type& data, int mi, int ni)
    {
        size_t offset = (ni + mi * MMAS_N * total_ksteps_) * BYTES_PER_ROW;
        fmha::ldg(data, ptr_ + offset);
    }

    inline __device__ void move()
    {
        // ptr_ += LOOP_STRIDE_BYTES;
        ptr_ += STEPQ * actual_seqlen_ * BYTES_PER_ELEMENT;
    }

    inline __device__ void move_k()
    {
        ptr_ += MMAS_N * BYTES_PER_ROW;
    }

    inline __device__ void move(int steps)
    {
        ptr_ += STEPQ * actual_seqlen_ * BYTES_PER_ELEMENT * steps;
    }

    // Move the pointer to the next store location.
    inline __device__ void move(int qstep_id, int kstep_id)
    {
        ptr_ = init_ptr_ + qstep_id * actual_seqlen_ * STEPQ * BYTES_PER_ELEMENT + kstep_id * MMAS_N * BYTES_PER_ROW;
    }

    char* ptr_;
    char* init_ptr_;
    int actual_seqlen_;
    int total_ksteps_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile, typename Base = Gmem_tile_mma_sd<Traits, Cta_tile, sizeof(uint16_t)>>
struct Gmem_tile_mma_s : public Base
{
    enum
    {
        M = Base::MMAS_M
    };

    enum
    {
        N = Base::MMAS_N
    };

    using Type = typename Base::Type;

    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_mma_s(Params const& params, Block_info const& block_info, int tidx)
        : Base(params.s_ptr, params, block_info, tidx)
    {
    }

    template <typename Mask>
    inline __device__ void store(float const (&softmax)[2 * M][4 * N], Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {

                float tmp00 = softmax[2 * mi + 0][4 * ni + 0];
                float tmp01 = softmax[2 * mi + 0][4 * ni + 1];
                float tmp02 = softmax[2 * mi + 0][4 * ni + 2];
                float tmp03 = softmax[2 * mi + 0][4 * ni + 3];

                float tmp10 = softmax[2 * mi + 1][4 * ni + 0];
                float tmp11 = softmax[2 * mi + 1][4 * ni + 1];
                float tmp12 = softmax[2 * mi + 1][4 * ni + 2];
                float tmp13 = softmax[2 * mi + 1][4 * ni + 3];

                uint4 dst;
                dst.x = fmha::float2_to_half2(tmp00, tmp01);
                dst.y = fmha::float2_to_half2(tmp02, tmp03);
                dst.z = fmha::float2_to_half2(tmp10, tmp11);
                dst.w = fmha::float2_to_half2(tmp12, tmp13);
                // Thread participates if any of the registers in the fragment is valid.
                if (mask.any_valid(mi, ni))
                {
                    Base::store(dst, mi, ni);
                }
            }
        }
    }

    // for debug, store the sign to evalue the correctness of dropout in the unit test
    template <typename Mask>
    inline __device__ void store_sign(float const (&softmax)[2 * M][4 * N], Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {

                float tmp00 = softmax[2 * mi + 0][4 * ni + 0];
                float tmp01 = softmax[2 * mi + 0][4 * ni + 1];
                float tmp02 = softmax[2 * mi + 0][4 * ni + 2];
                float tmp03 = softmax[2 * mi + 0][4 * ni + 3];

                float tmp10 = softmax[2 * mi + 1][4 * ni + 0];
                float tmp11 = softmax[2 * mi + 1][4 * ni + 1];
                float tmp12 = softmax[2 * mi + 1][4 * ni + 2];
                float tmp13 = softmax[2 * mi + 1][4 * ni + 3];

                tmp00 = tmp00 >= 0.f ? 1.0f : -1.0f;
                tmp01 = tmp01 >= 0.f ? 1.0f : -1.0f;
                tmp02 = tmp02 >= 0.f ? 1.0f : -1.0f;
                tmp03 = tmp03 >= 0.f ? 1.0f : -1.0f;
                tmp10 = tmp10 >= 0.f ? 1.0f : -1.0f;
                tmp11 = tmp11 >= 0.f ? 1.0f : -1.0f;
                tmp12 = tmp12 >= 0.f ? 1.0f : -1.0f;
                tmp13 = tmp13 >= 0.f ? 1.0f : -1.0f;

                uint4 dst;
                dst.x = fmha::float2_to_half2(tmp00, tmp01);
                dst.y = fmha::float2_to_half2(tmp02, tmp03);
                dst.z = fmha::float2_to_half2(tmp10, tmp11);
                dst.w = fmha::float2_to_half2(tmp12, tmp13);
                // Thread participates if any of the registers in the fragment is valid.
                if (mask.any_valid(mi, ni))
                {
                    Base::store(dst, mi, ni);
                }
            }
        }
    }

    template <typename Mask>
    inline __device__ void softmax_to_regs(float const (&softmax)[2 * M][4 * N], uint4 (&regs)[M][N], Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {

                float tmp00 = softmax[2 * mi + 0][4 * ni + 0];
                float tmp01 = softmax[2 * mi + 0][4 * ni + 1];
                float tmp02 = softmax[2 * mi + 0][4 * ni + 2];
                float tmp03 = softmax[2 * mi + 0][4 * ni + 3];

                float tmp10 = softmax[2 * mi + 1][4 * ni + 0];
                float tmp11 = softmax[2 * mi + 1][4 * ni + 1];
                float tmp12 = softmax[2 * mi + 1][4 * ni + 2];
                float tmp13 = softmax[2 * mi + 1][4 * ni + 3];

                regs[mi][ni] = make_uint4(0, 0, 0, 0);

                if (mask.any_valid(mi, ni))
                {
                    regs[mi][ni].x = fmha::float2_to_half2(tmp00, tmp01);
                    regs[mi][ni].y = fmha::float2_to_half2(tmp02, tmp03);
                    regs[mi][ni].z = fmha::float2_to_half2(tmp10, tmp11);
                    regs[mi][ni].w = fmha::float2_to_half2(tmp12, tmp13);
                }
            }
        }
    }

    template <typename Mask, typename Fragment>
    inline __device__ void store(Fragment const (&frag)[N][M], Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
                uint4 dst;
                dst.x = frag[ni][mi].reg(0);
                dst.y = frag[ni][mi].reg(2);
                dst.z = frag[ni][mi].reg(1);
                dst.w = frag[ni][mi].reg(3);

                // Thread participates if any of the registers in the fragment is valid.
                if (mask.any_valid(mi, ni))
                {
                    Base::store(dst, mi, ni);
                }
            }
        }
    }

    template <typename Mask>
    inline __device__ void load(uint4 (&regs)[M][N], Mask const& mask)
    {
#pragma unroll
        for (int mi = 0; mi < M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < N; ni++)
            {
                regs[mi][ni] = make_uint4(0, 0, 0, 0);
                // Thread participates if any of the registers in the fragment is valid.
                if (mask.any_valid(mi, ni))
                {
                    Base::load(regs[mi][ni], mi, ni);
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The base class.
    typename Base = fmha::v2::Gmem_tile_qkv<Traits, Cta_tile, Traits::BITS_PER_ELEMENT_A, Cta_tile::M, Cta_tile::K,
        Cta_tile::VALID_K, false, false>>
struct Gmem_tile_dout : public Base
{

    // The step in the Q dimension is 16.
    static_assert(Cta_tile::M == 16);
    // The step in the K/V dimension.
    static_assert(Cta_tile::K == 64 || Cta_tile::K == 128 || Cta_tile::K == 256);
    // Supported sizes per row.
    static_assert(Base::BYTES_PER_ROW == 128 || Base::BYTES_PER_ROW == 256 || Base::BYTES_PER_ROW == 512);

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_dout(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, 0, block_info, tidx)
    {

        this->qkv_ptr_ = reinterpret_cast<char*>(params.do_ptr);
        this->params_qkv_stride_in_bytes_ = params.o_stride_in_bytes; // needed for move

        // Compute the position of the thread in the row.
        int col = tidx % Base::THREADS_PER_ROW;

        // The row offset in the batched GEMM. For each seq element, we store O in that order.
        int64_t row_offset
            = (int64_t) this->row_ * params.o_stride_in_bytes + block_info.bidx * Base::VALID_BYTES_PER_ROW;

        // Assemble the final pointer.
        this->qkv_ptr_ += row_offset + col * Base::BYTES_PER_LDG;
        this->qkv_ptr_init_ = this->qkv_ptr_;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: Merge that class with Gmem_tile_dout. The only difference seems to be the pointer.

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The base class.
    typename Base = fmha::v2::Gmem_tile_qkv<Traits, Cta_tile, Traits::BITS_PER_ELEMENT_A, Cta_tile::M, Cta_tile::K,
        Cta_tile::VALID_K, false, false>>
struct Gmem_tile_out : public Base
{

    // The step in the Q dimension is 16.
    static_assert(Cta_tile::M == 16);
    // The step in the K/V dimension.
    static_assert(Cta_tile::K == 64 || Cta_tile::K == 128 || Cta_tile::K == 256);
    // Supported sizes per row.
    static_assert(Base::BYTES_PER_ROW == 128 || Base::BYTES_PER_ROW == 256 || Base::BYTES_PER_ROW == 512);

    enum
    {
        LDGS = Base::LDGS
    };

    enum
    {
        ROWS_PER_LDG = Base::ROWS_PER_LDG
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_out(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, 0, block_info, tidx)
    {

        this->qkv_ptr_ = reinterpret_cast<char*>(params.o_ptr);
        this->params_qkv_stride_in_bytes_ = params.o_stride_in_bytes; // needed for move

        // Compute the position of the thread in the row.
        int col = tidx % Base::THREADS_PER_ROW;

        // The row offset in the batched GEMM. For each seq element, we store O in that order.
        int64_t row_offset
            = (int64_t) this->row_ * params.o_stride_in_bytes + block_info.bidx * Base::VALID_BYTES_PER_ROW;

        // Assemble the final pointer.
        this->qkv_ptr_ += row_offset + col * Base::BYTES_PER_LDG;
        this->qkv_ptr_init_ = this->qkv_ptr_;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, typename Cta_tile>
struct Gmem_softmax_sum
{

    // The helper class.
    using Helper = fmha::v2::Gmem_tile_qkv<Traits, Cta_tile, Traits::BITS_PER_ELEMENT_A, Cta_tile::M, Cta_tile::K,
        Cta_tile::VALID_K, false, false>;
    // The Mma tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    enum
    {
        MMAS_M = Mma_tile::MMAS_M
    };

    // The size of each element.
    enum
    {
        BYTES_PER_ELEMENT = 4
    };

    // The number of bytes per element. TODO: It is not arch independent. Fix me!!!
    enum
    {
        BYTES_PER_MMA = (Cta_tile::THREADS_PER_WARP / 4) * 2 * BYTES_PER_ELEMENT
    };

    // The number of rows produced by iteration (STEPQ).
    enum
    {
        ROWS = Cta_tile::M
    };

    // The number of rows written per LDG.
    enum
    {
        ROWS_PER_LDG = Helper::ROWS_PER_LDG
    };

    // The number of LDGs.
    enum
    {
        LDGS = Helper::LDGS
    };

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_softmax_sum(Params const& params, int const tidx)
        : ptr_(reinterpret_cast<char*>(params.softmax_sum_ptr))
        , tidx_(tidx)
    {

        // The block index for the batch.
        int const bidb = blockIdx.y;
        // The block index for the head.
        int const bidh = blockIdx.x;
        // The block index.
        // size_t bidx = bidb * params.h + bidh;
        uint32_t bidx = bidb * params.h + bidh;

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // The distance between two blocks (in bytes).
        // size_t block_stride_bytes = params.seqlen_q * BYTES_PER_ELEMENT;
        uint32_t block_stride_bytes = params.s * BYTES_PER_ELEMENT;

        // Set store location for each thread at the beginning of the loop
        ptr_row_ = ptr_ + bidx * block_stride_bytes;
        ptr_ += bidx * block_stride_bytes + (lane / 4) * BYTES_PER_ELEMENT;
        init_ptr_row_ = ptr_row_;
        init_ptr_ = ptr_;
    }

    // Store data to global memory.
    inline __device__ void store(uint32_t const (&data)[MMAS_M * 2])
    {
        // Rematerialize the position in the warp.
        int warp = tidx_ / Cta_tile::THREADS_PER_WARP;
        int lane = tidx_ % Cta_tile::THREADS_PER_WARP;

        // Only the 1st thread of the 1st warp stores the sum for Softmax.
        if ((warp == 0) && (lane % 4 == 0))
        {
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
                // TODO: Not sure if it's right for MMAS_M > 1
                fmha::stg(ptr_ + mi * BYTES_PER_MMA + 0 * BYTES_PER_ELEMENT, data[mi * 2 + 0]);
                fmha::stg(ptr_ + mi * BYTES_PER_MMA + 8 * BYTES_PER_ELEMENT, data[mi * 2 + 1]);
            }
        }
    }

    // Store data to global memory.
    inline __device__ void store_row(uint32_t const (&data)[LDGS], int const row)
    {
        enum
        {
            BYTES_PER_LDG = ROWS_PER_LDG * BYTES_PER_ELEMENT
        };

#pragma unroll
        for (int mi = 0; mi < LDGS; ++mi)
        {
            // TODO: Not sure if it's right for MMAS_M > 1
            int64_t offset = (int64_t) mi * BYTES_PER_LDG + row * BYTES_PER_ELEMENT;
            fmha::stg(ptr_row_ + offset, data[mi]);
        }
    }

    // Load from global memory.
    inline __device__ void load(uint32_t (&data)[MMAS_M * 2])
    {
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            // TODO: Not sure if it's right for MMAS_M > 1
            fmha::ldg(data[mi * 2 + 0], ptr_ + mi * BYTES_PER_MMA + 0 * BYTES_PER_ELEMENT);
            fmha::ldg(data[mi * 2 + 1], ptr_ + mi * BYTES_PER_MMA + 8 * BYTES_PER_ELEMENT);
        }
    }

    // Load from global memory.
    inline __device__ void load_next(uint32_t (&data)[MMAS_M * 2], int move_steps = 1)
    {
        char* ptr_next = ptr_ + move_steps * ROWS * BYTES_PER_ELEMENT;
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
            // TODO: Not sure if it's right for MMAS_M > 1
            fmha::ldg(data[mi * 2 + 0], ptr_next + mi * BYTES_PER_MMA + 0 * BYTES_PER_ELEMENT);
            fmha::ldg(data[mi * 2 + 1], ptr_next + mi * BYTES_PER_MMA + 8 * BYTES_PER_ELEMENT);
        }
    }

    template <int N>
    inline __device__ void load_row(uint32_t (&data)[LDGS], int const row)
    {
        enum
        {
            BYTES_PER_LDG = ROWS_PER_LDG * BYTES_PER_ELEMENT
        };

#pragma unroll
        for (int ni = 0; ni < LDGS; ++ni)
        {
            fmha::ldg(data[ni], ptr_row_ + ni * BYTES_PER_LDG + row * BYTES_PER_ELEMENT);
        }
    }

    // Move the pointer to the next location.
    inline __device__ void move()
    {
        ptr_ += ROWS * BYTES_PER_ELEMENT;
        ptr_row_ += ROWS * BYTES_PER_ELEMENT;
    }

    // Move the pointer to the next location.
    inline __device__ void move(int const steps)
    {
        ptr_ += ROWS * BYTES_PER_ELEMENT * steps;
        ptr_row_ += ROWS * BYTES_PER_ELEMENT * steps;
    }

    // Move the pointer to a specific location.
    inline __device__ void move_to(int const steps)
    {
        ptr_ = init_ptr_ + ROWS * BYTES_PER_ELEMENT * steps;
        ptr_row_ = init_ptr_row_ + ROWS * BYTES_PER_ELEMENT * steps;
    }

    // The pointer.
    char *ptr_, *init_ptr_;
    char *ptr_row_, *init_ptr_row_;
    int const tidx_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // Are the heads in qkv interleaved?
    bool HEADS_INTERLEAVED,
    // The base class.
    typename Base = fmha::v2::Gmem_tile_o<Traits, Cta_tile, 1>>
struct Gmem_tile_dq : public Base
{

    // DEBUG: We do not support interleaved heads for that case.
    // static_assert(!HEADS_INTERLEAVED, "");
    // END OF DEBUG.

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_dq(Params const& params, Block_info const& block_info, int tidx)
        : Base(params, block_info, tidx)
    {

        this->o_ptr_ = reinterpret_cast<char*>(params.dqkv_ptr);
        this->params_o_stride_in_bytes_ = params.qkv_stride_in_bytes; // needed for move

        // Compute the position of the thread in the row.
        int col = tidx % Base::THREADS_PER_ROW;

        // The row offset in the batched GEMM. For each seq element, we store O in that order.
        int64_t row_offset = (int64_t) this->row_ * params.qkv_stride_in_bytes;

        if (HEADS_INTERLEAVED)
        {
            row_offset += block_info.bidx * 3 * Base::VALID_BYTES_PER_ROW;
        }
        else
        {
            // The 1st position of Q in the batch.
            int64_t batch_idx = block_info.sum_s * 3 /* + 0, because it's "Q" */;
            // The offset of the row.
            row_offset += (batch_idx * params.h + block_info.bidh) * Base::VALID_BYTES_PER_ROW;
        }

        // Assemble the final pointer.
        this->o_ptr_ += row_offset + col * Base::BYTES_PER_STG;

        this->init_row_ = this->row_;
        this->init_o_ptr_ = this->o_ptr_;
    }

    inline __device__ void reset()
    {
        this->o_ptr_ = this->init_o_ptr_;
        this->row_ = this->init_row_;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_, typename Cta_tile_>
struct Gmem_tile_dq_red_base
{

    // The instruction traits.
    using Traits = Traits_;
    // The CTA descriptor.
    using Cta_tile = Cta_tile_;
    // The associated MMA tile.
    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    // DEBUG: Make sure we have only one warp in the M/N dimensions of the tile.
    static_assert(Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 1, "");

    // END OF DEBUG.

    // The number of bytes per element is 4 (fp32).
    enum
    {
        BYTES_PER_ELEMENT = 4
    };

    // The number of elements per row.
    enum
    {
        ELEMENTS_PER_ROW = Cta_tile::N
    };

    // The number of threads per row when storing to global memory. We use RED.32.
    enum
    {
        THREADS_PER_ROW = ELEMENTS_PER_ROW
    };

    // The size of a row in bytes.
    enum
    {
        BYTES_PER_ROW = ELEMENTS_PER_ROW * BYTES_PER_ELEMENT
    };

    // The number of rows in smem. We process MMA rows one-by-one to reduce the footprint in smem.
    enum
    {
        ROWS = Mma_tile::M_PER_MMA_PER_CTA
    };

    // The number of rows per read per RED.
    enum
    {
        ROWS_PER_RED = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW
    };

    // Make sure we do not have partial stores.
    static_assert(ROWS % ROWS_PER_RED == 0, "");

    // The number of REDs per thread/CTA.
    enum
    {
        REDS = ROWS / ROWS_PER_RED
    };

    // The valid number of threads per row. Note that it >= actual head_dim.
    enum
    {
        VALID_THREADS_PER_ROW = Mma_tile::VALID_MMAS_N * Mma_tile::N_PER_MMA
    };

    // Ctor.
    template <typename Params, typename Binfo_>
    inline __device__ Gmem_tile_dq_red_base(Params const& params, Binfo_ const& binfo, int tidx)
        : ptr_(reinterpret_cast<float*>(params.dq_acc_ptr))
        , next_seq_offset_factor_(binfo.next_seq_offset_factor)
    {

        // Compute the offset.
        int row = tidx / THREADS_PER_ROW;
        col_ = tidx % THREADS_PER_ROW;

        // We store the data as heads x total_sequences x head_dim.

        // Compute the offset. bidh is the head, total_s is the sum of sequence lengths and sum_s
        // is the offset to the beginning of the sequence written by that CTA.
        int64_t offset = (int64_t) binfo.bidh * params.total_s + binfo.sum_s;

        // Update the base pointer (the pointer has type float*).
        ptr_ += offset * Cta_tile::N + col_;

        // Keep track of the token to be able to detect OOB accesses.
        token_ = row;
        // The sequence length.
        seqlen_ = binfo.actual_seqlen;
    }

    // Store the data to global memory.
    inline __device__ void store(int step_qi, int mi, int ii, float out)
    {
        // Compute the offset in the dQ dimension.
        int token = token_ + step_qi * Cta_tile::M + mi * Mma_tile::M_PER_MMA + ii * ROWS_PER_RED;

        // Store if the token is in the sequence.
        if (token < seqlen_ && col_ < VALID_THREADS_PER_ROW)
        {
            atomicAdd(&ptr_[token * next_seq_offset_factor_ * Cta_tile::N], out);
        }
    }

    // The pointer.
    float* ptr_;
    // The token.
    int token_;
    // The length of the sequence.
    int seqlen_;
    // The col.
    int col_;
    // The next sequence's offset factor
    int next_seq_offset_factor_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_, typename Cta_tile_>
struct Gmem_tile_dq_red
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile_>
struct Gmem_tile_dq_red<fmha::Ampere_hmma_fp32_traits, Cta_tile_>
    : public Gmem_tile_dq_red_base<fmha::Ampere_hmma_fp32_traits, Cta_tile_>
{
    // The instruction traits.
    using Traits = fmha::Ampere_hmma_fp32_traits;
    // The base class.
    using Base = Gmem_tile_dq_red_base<Traits, Cta_tile_>;

    // Ctor.
    template <typename Params, typename Binfo_>
    inline __device__ Gmem_tile_dq_red(Params const& params, Binfo_ const& binfo, int tidx)
        : Base(params, binfo, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile_>
struct Gmem_tile_dq_red<fmha::Ampere_hmma_bf16_traits, Cta_tile_>
    : public Gmem_tile_dq_red_base<fmha::Ampere_hmma_bf16_traits, Cta_tile_>
{
    // The instruction traits.
    using Traits = fmha::Ampere_hmma_bf16_traits;
    // The base class.
    using Base = Gmem_tile_dq_red_base<Traits, Cta_tile_>;

    // Ctor.
    template <typename Params, typename Binfo_>
    inline __device__ Gmem_tile_dq_red(Params const& params, Binfo_ const& binfo, int tidx)
        : Base(params, binfo, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
