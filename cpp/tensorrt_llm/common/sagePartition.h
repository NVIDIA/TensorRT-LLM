/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <cute/tensor.hpp>

namespace tensorrt_llm::common
{

// How SageAttention groups tokens into quantization scales, as a CuTe layout.
//
// A partition is a bijective layout ((intra...), (group...)) -> token offset inside one *tile*:
//   mode 0 enumerates the tokens that share one scale,
//   mode 1 enumerates the scale groups of a tile,
//   size(layout) is the tile size in tokens.
//
// This is what differs between GPUs. SM100 wants each scale to cover a run of contiguous tokens,
// so its tile is one group. SM90 scales per thread, and the BMM1 fragment hands each thread a
// *strided* set of rows/columns, so its groups interleave and the group index inside a tile is
// swizzled. Expressing both as a layout lets one quantizer serve both: it just evaluates
// partition(i, g).
//
//   ScaleAlign is the alignment, in scales, that the consumer needs a sequence's scale base to
//   have. The SM90 K path loads its four scales as one float4, so it needs 4; everything else
//   needs 1.
//
//   HeadStrideCountsBlocks selects how the consumer derives the head stride of a scale buffer.
//   The two conventions agree on where a sequence's scales start and differ only in the total,
//   so a partition has to declare the one its consumer was built with.

// -------------------------------------------------------------------------------------------
// SM100 / contiguous: a tile is one group of BlockSize consecutive tokens.
template <int BlockSize>
struct ContiguousPartition
{
    using Layout = cute::Layout<cute::Shape<cute::Shape<cute::Int<BlockSize>>, cute::Shape<cute::_1>>,
        cute::Stride<cute::Stride<cute::_1>, cute::Stride<cute::Int<BlockSize>>>>;
    static constexpr int ScaleAlign = 1;
    // Head stride: started blocks over the whole batch, plus one spare per extra sequence.
    static constexpr bool HeadStrideCountsBlocks = true;
};

// -------------------------------------------------------------------------------------------
// SM90 Q: STEP_Q = 64 rows per tile, 2 rows per scale, 32 scales per tile.
//
// The BMM1 fragment gives the thread at (warp w, lane l) the query rows quad_row and quad_row + 8,
// with quad_row = 16 * w + l / 4. Numbering the groups by pair = w * 8 + l / 4 (which is how
// fmha/warpspec/compute.h indexes them), group pair = p8 + 8 * w owns rows 16 * w + p8 + 8 * r:
//   intra  : (r)      stride 8
//   group  : (p8, w)  strides (1, 16), linear index p8 + 8 * w == pair
struct HopperQPartition
{
    using Layout = cute::Layout<cute::Shape<cute::Shape<cute::_2>, cute::Shape<cute::_8, cute::_4>>,
        cute::Stride<cute::Stride<cute::_8>, cute::Stride<cute::_1, cute::_16>>>;
    static constexpr int ScaleAlign = 1;
    // Head stride: the sequence base function evaluated at (sumSeqLens, batchSize).
    static constexpr bool HeadStrideCountsBlocks = false;
};

// -------------------------------------------------------------------------------------------
// SM90 K: STEP_KV = 256 keys per tile, 16 keys per scale, 16 scales per tile.
//
// A thread with lane % 4 == quad owns the key columns 8 * ni + 2 * quad + e, ni in [0, 32),
// e in {0, 1}. Splitting that fragment into 4 sub-fragments by ni-range gives 16 keys per scale;
// writing ni = 8 * g + n, a key is 64 * g + 8 * n + 2 * quad + e:
//   intra  : (e, n)     strides (1, 8)
//   group  : (g, quad)  strides (64, 2), linear index g + 4 * quad == quad * K_CHUNKS + g
struct HopperKPartition
{
    using Layout = cute::Layout<cute::Shape<cute::Shape<cute::_2, cute::_8>, cute::Shape<cute::_4, cute::_4>>,
        cute::Stride<cute::Stride<cute::_1, cute::_8>, cute::Stride<cute::_64, cute::_2>>>;
    // The consumer loads a thread's four chunk scales with one 128-bit access.
    static constexpr int ScaleAlign = 4;
    // Head stride: the sequence base function evaluated at (sumSeqLens, batchSize).
    static constexpr bool HeadStrideCountsBlocks = false;
};

// -------------------------------------------------------------------------------------------
// Derived geometry, shared by the kernel and the host.
template <typename Partition>
struct PartitionTraits
{
    using Layout = typename Partition::Layout;

    // Tokens sharing one scale.
    static constexpr int TokensPerScale = cute::size<0>(Layout{});
    // Scales in one tile.
    static constexpr int ScalesPerTile = cute::size<1>(Layout{});
    // Tokens in one tile.
    static constexpr int TileTokens = cute::size(Layout{});
    static constexpr int ScaleAlign = Partition::ScaleAlign;

    static_assert(TileTokens == TokensPerScale * ScalesPerTile, "partition must be a bijection");
    static_assert(cute::cosize(Layout{}) == TileTokens, "partition must tile densely");
    static_assert(ScaleAlign > 0 && (ScaleAlign & (ScaleAlign - 1)) == 0, "ScaleAlign must be 2^k");

    // A sequence reserves one spare tile's worth of scales, because a partial final tile still
    // indexes all ScalesPerTile of them, plus ScaleAlign - 1 to absorb the base round-up. Rounded
    // up to a multiple of ScaleAlign so that every base stays aligned.
    static constexpr int SlotsPerSeq = ((ScalesPerTile + ScaleAlign - 1) + ScaleAlign - 1) / ScaleAlign * ScaleAlign;

    // Where sequence seqIdx's scales start within a head.
    CUTE_HOST_DEVICE static constexpr int scaleBase(int cuSeqLen, int seqIdx)
    {
        return ((cuSeqLen / TokensPerScale + ScaleAlign - 1) / ScaleAlign) * ScaleAlign + seqIdx * SlotsPerSeq;
    }

    // Scales per head, i.e. the head stride of the scale buffer. Depends only on the batch size and
    // the total token count, never on the longest sequence. The consumer kernels are not passed
    // this value; they recompute it, so it has to match the convention they were built with.
    CUTE_HOST_DEVICE static constexpr int scaleHeadStride(int sumSeqLens, int batchSize)
    {
        if constexpr (Partition::HeadStrideCountsBlocks)
        {
            return (sumSeqLens + TokensPerScale - 1) / TokensPerScale + batchSize - 1;
        }
        else
        {
            return scaleBase(sumSeqLens, batchSize);
        }
    }

    // Token index inside a tile for intra-group element i of group g.
    CUTE_HOST_DEVICE static constexpr int tokenInTile(int i, int g)
    {
        return Layout{}(i, g);
    }
};

} // namespace tensorrt_llm::common
