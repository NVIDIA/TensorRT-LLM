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

#include "tensorrt_llm/common/config.h"

#if defined(__CUDACC__)
#define TLLM_SAGE_HOST_DEVICE __host__ __device__
#else
#define TLLM_SAGE_HOST_DEVICE
#endif

TRTLLM_NAMESPACE_BEGIN

namespace common
{

// How Q or K tokens are grouped into scales. See sagePartition.h for the token layouts.
enum class SageScalePartition
{
    Contiguous = 0, //!< tokenBlockSize consecutive tokens share a scale.
    HopperQ,        //!< Two rows eight apart share a scale, with 32 scales per 64-row tile.
    HopperK,        //!< Sixteen strided keys share a scale, with 16 scales per 256-key tile.
};

// Scale-buffer geometry shared by host sizing and device partition traits.
template <int TokensPerScale, int ScalesPerTile, int ScaleAlign, bool HeadStrideCountsBlocks>
struct SageScaleGeometry
{
    static constexpr int kTokensPerScale = TokensPerScale;
    static constexpr int kScalesPerTile = ScalesPerTile;
    static constexpr int kScaleAlign = ScaleAlign;
    static_assert(ScaleAlign > 0 && (ScaleAlign & (ScaleAlign - 1)) == 0);

    // Reserve one spare tile per sequence and space for rounding its base up to ScaleAlign.
    static constexpr int kSlotsPerSeq = ((ScalesPerTile + ScaleAlign - 1) + ScaleAlign - 1) / ScaleAlign * ScaleAlign;

    TLLM_SAGE_HOST_DEVICE static constexpr int scaleBase(int cuSeqLen, int seqIdx)
    {
        return ((cuSeqLen / TokensPerScale + ScaleAlign - 1) / ScaleAlign) * ScaleAlign + seqIdx * kSlotsPerSeq;
    }

    // Contiguous consumers count started blocks over the batch; Hopper consumers derive the
    // head stride from the sequence base at the total token count and batch size.
    TLLM_SAGE_HOST_DEVICE static constexpr int scaleHeadStride(int sumSeqLens, int batchSize)
    {
        if constexpr (HeadStrideCountsBlocks)
        {
            return (sumSeqLens + TokensPerScale - 1) / TokensPerScale + batchSize - 1;
        }
        else
        {
            return scaleBase(sumSeqLens, batchSize);
        }
    }
};

template <int BlockSize>
using ContiguousScaleGeometry = SageScaleGeometry<BlockSize, 1, 1, true>;
using HopperQScaleGeometry = SageScaleGeometry<2, 32, 1, false>;
using HopperKScaleGeometry = SageScaleGeometry<16, 16, 4, false>;

inline SageScalePartition getSageQPartition(bool perThread)
{
    return perThread ? SageScalePartition::HopperQ : SageScalePartition::Contiguous;
}

inline SageScalePartition getSageKPartition(bool perThread)
{
    return perThread ? SageScalePartition::HopperK : SageScalePartition::Contiguous;
}

// Scales per head, including spare sequence tiles. Returns zero for empty shapes, disabled
// quantization (tokenBlockSize <= 0), or a partition/block-size pair without a kernel instantiation.
int getSageScaleHeadStride(SageScalePartition partition, int tokenBlockSize, int sumSeqLens, int batchSize);

} // namespace common

TRTLLM_NAMESPACE_END

#undef TLLM_SAGE_HOST_DEVICE
