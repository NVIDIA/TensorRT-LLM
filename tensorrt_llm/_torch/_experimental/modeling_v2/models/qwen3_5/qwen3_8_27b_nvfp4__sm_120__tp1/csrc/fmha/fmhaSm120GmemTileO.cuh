/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Target-private copy of the register-direct O store of the SM120 e4m3 head-256 context FMHA.
//
// Gmem_tile_o_16bit_from_regs and Gmem_tile_o_from_regs_selector: an O tile that stores straight from registers,
// next to the stock ones in cpp/kernels/fmha_v2/src/fmha/gmem_tile_o_packed.h. Namespace TRTLLM_MV2_NS::fmha_sm120;
// the stock names it uses are qualified
// (fmha::Uint_from_size_in_bytes, fmha::v2::Acc_packer, fmha::v2::Gmem_tile_o_{bfloat16,uint16}). Everything else is
// the stock (unmodified) fmha_v2.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <fmha/gmem_tile_o_packed.h>
#include <fmha/kernel_traits.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

#include <type_traits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace fmha_sm120
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Stores the fp32 BMM2 accumulators of Ada_qmma_e4m3_fp32_traits as 16-bit values straight from the
// registers, without the shared-memory round trip Smem_tile_o::store -> Smem_tile_o::load ->
// Gmem_tile_o_16bit::store.
//
// It is only used where that round trip moves data without any arithmetic: 4x1x1 warps (no split-K) and
// one MMA per warp in M (Cta_tile::M == 64). There, for the thread (warp w, lane l), quad q = l % 4 and the
// MMA ni, the round trip (the XOR swizzle of the smem tile is undone by its reader) writes
//   acc[0][ni] regs {0, 4, 1, 5} to row 16 * w + l / 4,     cols 16 * ni + 4 * q + {0, 1, 2, 3},
//   acc[0][ni] regs {2, 6, 3, 7} to row 16 * w + l / 4 + 8, cols 16 * ni + 4 * q + {0, 1, 2, 3}.
// This class writes the same values (Acc_packer: x * scale_bmm2, then the conversion to 16 bit) to the same
// addresses, with the same row guard (row < actual_q_seqlen) and the same column guard (8-byte chunk index
// < dv / 4). The output is bitwise identical and the CTA needs no shared memory for O.
template <typename Traits, typename Cta_tile, typename Output_type>
struct Gmem_tile_o_16bit_from_regs
{

    static_assert(std::is_same<Traits, fmha::Ada_qmma_e4m3_fp32_traits>::value);
    static_assert(std::is_same<Output_type, uint16_t>::value || std::is_same<Output_type, nv_bfloat16>::value);
    static_assert(Cta_tile::WARPS_M == 4 && Cta_tile::WARPS_N == 1 && Cta_tile::WARPS_K == 1, "");
    static_assert(Cta_tile::M == 64 && Cta_tile::VALID_N == Cta_tile::N, "");

    using Mma_tile = typename Traits::template Mma_tile<Cta_tile>;

    static_assert(Mma_tile::MMAS_M == 1 && Mma_tile::M_PER_MMA == 16 && Mma_tile::N_PER_MMA == 16, "");

    // The size of each element.
    static constexpr int BYTES_PER_ELEMENT = 2;

    // The number of rows of O written by the tile.
    static constexpr int ROWS = Cta_tile::M;

    // The size of each STG (4 elements).
    static constexpr int BYTES_PER_STG = 8;

    // The STG packed data type
    using Stg_packed_type = typename fmha::Uint_from_size_in_bytes<BYTES_PER_STG>::Type;

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Gmem_tile_o_16bit_from_regs(Params const& params, Block_info const& block_info, int tidx,
        int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
        : params_o_stride_in_bytes_(params.o_stride_in_bytes)
        , actual_seqlen_(block_info.actual_q_seqlen)
        , params_scale_bmm2_(params.scale_bmm2_d ? *params.scale_bmm2_d : params.scale_bmm2)
        , o_ptr_(reinterpret_cast<char*>(params.o_ptr))
    {

        // The row of the accumulator regs {0, 4, 1, 5}: (tidx & 0x60) / 2 + (tidx & 0x1c) / 4.
        int row = (tidx / 32) * Mma_tile::M_PER_MMA + (tidx % 32) / 4;
        // The quad position of the thread: its 8-byte chunk within the 16 columns of one MMA.
        quad_ = tidx % 4;

        // Store the row to check against the length before stores.
        row_ = cta_row_offset + row;
        int col_in_bytes = cta_col_offset_in_bytes + quad_ * BYTES_PER_STG;

        // The row offset in the batched GEMM (same arithmetic as Gmem_tile_o_16bit).
        int64_t row_offset = (int64_t) row_ * params.o_stride_in_bytes;
        // The amount of bytes per row without padding (runtime).
        int const valid_bytes_per_row = params.dv * BYTES_PER_ELEMENT;
        // Take the batch/head offset into account.
        row_offset += block_info.bidx * valid_bytes_per_row;
        // Assemble the final pointer.
        o_ptr_ += row_offset + col_in_bytes;

        // Do not store the chunks in the padded area.
        valid_chunks_ = valid_bytes_per_row / BYTES_PER_STG;
    }

    // Store the accumulators.
    template <typename Accumulator, int M, int N>
    inline __device__ void store(Accumulator const (&acc)[M][N]) const
    {
        static_assert(M == 1 && N == Mma_tile::MMAS_N, "");
        using Src_type = typename Traits::Accumulator_type;

#pragma unroll
        for (int hi = 0; hi < 2; ++hi)
        {
            // Break early if we exceed s_i...
            if (row_ + hi * 8 >= actual_seqlen_)
            {
                return;
            }
#pragma unroll
            for (int ni = 0; ni < N; ++ni)
            {
                // The same 4 registers as Regs_to_rows_8bit puts in one 16B row chunk of Smem_tile_o.
                uint4 src;
                src.x = acc[0][ni].reg(hi * 2 + 0);
                src.y = acc[0][ni].reg(hi * 2 + 4);
                src.z = acc[0][ni].reg(hi * 2 + 1);
                src.w = acc[0][ni].reg(hi * 2 + 5);
                // Packs the 32bit values to 16bit, with the extra scaling by scale_bmm2.
                Stg_packed_type dst = fmha::v2::Acc_packer<Src_type, Output_type>::run(this, src);
                // Store the result.
                if (ni * (Mma_tile::N_PER_MMA / 4) + quad_ < valid_chunks_)
                {
                    fmha::stg(
                        o_ptr_ + hi * 8 * params_o_stride_in_bytes_ + ni * Mma_tile::N_PER_MMA * BYTES_PER_ELEMENT,
                        dst);
                }
            }
        }
    }

    // The stride between rows for the O matrix.
    int64_t params_o_stride_in_bytes_;
    // The size of the sequence length computed by that CTA.
    int actual_seqlen_;
    // The scaling factor applied to the accumulators (read by Acc_packer).
    uint32_t const params_scale_bmm2_;
    // The pointer.
    char* o_ptr_;
    // The row stored by this thread (i.e. the position in that sequence) for regs {0, 4, 1, 5}.
    int row_;
    // The quad position of the thread.
    int quad_;
    // The number of valid 8-byte chunks per row.
    int valid_chunks_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Maps a Gmem_tile_o type to its register-direct variant (VALUE = false when there is none).
template <typename Gmem_tile_o>
struct Gmem_tile_o_from_regs_selector
{
    static constexpr bool VALUE = false;
    using Type = Gmem_tile_o;
};

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o_from_regs_selector<fmha::v2::Gmem_tile_o_bfloat16<Traits, Cta_tile, CTAS_PER_HEAD>>
{
    static constexpr bool VALUE = CTAS_PER_HEAD == 1;
    using Type = Gmem_tile_o_16bit_from_regs<Traits, Cta_tile, nv_bfloat16>;
};

template <typename Traits, typename Cta_tile, int CTAS_PER_HEAD>
struct Gmem_tile_o_from_regs_selector<fmha::v2::Gmem_tile_o_uint16<Traits, Cta_tile, CTAS_PER_HEAD>>
{
    static constexpr bool VALUE = CTAS_PER_HEAD == 1;
    using Type = Gmem_tile_o_16bit_from_regs<Traits, Cta_tile, uint16_t>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha_sm120

TRTLLM_MV2_NAMESPACE_END
