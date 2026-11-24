/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/attentionWorkspace.h"
#include "tensorrt_llm/common/sageScaleGeometry.h"

TRTLLM_NAMESPACE_BEGIN

namespace common::op
{

AttentionContextWorkspaceSizes AttentionWorkspaceManager::buildContextSizes(
    AttentionContextWorkspaceParams const& params) noexcept
{
    AttentionContextWorkspaceSizes sizes{};
    auto const queryHiddenUnits = params.numQueryHeads * params.headSize;
    auto const kvHiddenUnits = params.numAttnKvHeads * params.headSize;
    if (!params.enableContextFmha)
    {
        auto const paddedQueryTokens = params.batchSize * params.querySequenceLength;
        auto const paddedKvTokens = params.batchSize * params.kvSequenceLength;
        auto const scoreElements = paddedQueryTokens * params.numQueryHeads * params.kvSequenceLength;
        sizes.attentionMask = params.elementSize * paddedQueryTokens * params.kvSequenceLength;
        sizes.qBuf = params.elementSize * paddedQueryTokens * queryHiddenUnits;
        sizes.kBuf = params.elementSize * paddedKvTokens * kvHiddenUnits;
        sizes.vBuf = sizes.kBuf;
        sizes.qkBuf = params.elementSize * scoreElements;
        sizes.qkvBuf = sizes.qBuf;
        sizes.qkFloatBuf = sizeof(float) * scoreElements;
        sizes.paddingOffset = sizeof(int) * paddedQueryTokens;
        sizes.encoderPaddingOffset = sizeof(int) * params.batchSize * params.crossKvLength;
    }
    else
    {
        if (params.separateQAndKvInput)
        {
            sizes.qBuf = (params.fp8ContextFmha ? 1 : params.elementSize) * params.packedTokenCount * queryHiddenUnits;
            if (params.sageAttention)
            {
                sizes.fp8QBuf = params.packedTokenCount * queryHiddenUnits;
                sizes.fp8KBuf = params.totalKvTokenCount * kvHiddenUnits;
                sizes.fp8VBuf = sizes.fp8KBuf;
            }
        }
        else if (params.fp8ContextFmha)
        {
            sizes.fp8QkvBuf = params.packedTokenCount * (queryHiddenUnits + 2ULL * kvHiddenUnits);
        }
        sizes.fmhaTileCounter = sizeof(uint32_t);
    }

    sizes.cuQSeqlens = sizeof(int) * (params.batchSize + 1);
    sizes.cuKvSeqlens = sizes.cuQSeqlens;
    sizes.cuMaskRows = sizes.cuQSeqlens;
    sizes.rotaryInvFreq = sizeof(float) * params.batchSize * params.rotaryEmbeddingDim / 2;
    sizes.fmhaBmm1Scale = params.fp8ContextFmha ? sizeof(float) * 2 : 0;
    sizes.fmhaBmm2Scale = params.fp8ContextFmha ? sizeof(float) : 0;

    auto const qScaleStride = common::getSageScaleHeadStride(common::getSageQPartition(params.hopperSage),
        params.sageQBlockSize, static_cast<int>(params.packedTokenCount), static_cast<int>(params.batchSize));
    auto const kScaleStride = common::getSageScaleHeadStride(common::getSageKPartition(params.hopperSage),
        params.sageKBlockSize, static_cast<int>(params.totalKvTokenCount), static_cast<int>(params.batchSize));
    auto const vScaleCount
        = params.sageVBlockSize > 0 ? (kvHiddenUnits + params.sageVBlockSize - 1) / params.sageVBlockSize : 0;
    sizes.sageQScale = sizeof(float) * params.numQueryHeads * static_cast<size_t>(qScaleStride);
    sizes.sageKScale = sizeof(float) * params.numAttnKvHeads * static_cast<size_t>(kScaleStride);
    sizes.sageVScale = sizeof(float) * static_cast<size_t>(vScaleCount);
    sizes.sageKMean = params.sageSmoothK ? sizeof(float) * kvHiddenUnits : 0;
    sizes.sageKMeanSum = params.sageSmoothK ? sizeof(int64_t) * kvHiddenUnits : 0;

    if (params.sparseFmha)
    {
        constexpr size_t kRowsPerCta = 256;
        constexpr size_t kStatsPerRow = 2;
        constexpr size_t kPartialOElementSize = 2;
        auto const maxRows = kRowsPerCta * static_cast<size_t>(params.multiProcessorCount);
        sizes.fmhaMultiCtasKvScratch
            = maxRows * (sizeof(float) * kStatsPerRow + kPartialOElementSize * params.headSize);
    }
    sizes.tokensInfo = sizeof(int2) * params.packedTokenCount;
    return sizes;
}

} // namespace common::op

TRTLLM_NAMESPACE_END
