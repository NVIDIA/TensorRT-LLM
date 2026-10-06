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

TRTLLM_NAMESPACE_BEGIN

namespace common::op
{

AttentionContextWorkspaceSizes AttentionWorkspaceManager::buildUnfusedContextSizes(
    AttentionUnfusedContextWorkspaceParams const& params) noexcept
{
    AttentionContextWorkspaceSizes sizes{};
    if (params.packedTokenCount == 0)
    {
        sizes.cublasWorkspace = 0;
        return sizes;
    }

    auto const paddedQueryTokens = params.batchSize * params.querySequenceLength;
    auto const paddedKvTokens = params.batchSize * params.kvSequenceLength;
    auto const queryHiddenUnits = params.numAttnHeads * params.headSize;
    auto const kvHiddenUnits = params.numAttnKvHeads * params.headSize;
    auto const scoreElements = paddedQueryTokens * params.numHeads * params.kvSequenceLength;
    sizes.attentionMask = params.elementSize * paddedQueryTokens * params.kvSequenceLength;
    sizes.cuQSeqlens = sizeof(int) * (params.batchSize + 1);
    sizes.cuKvSeqlens = sizes.cuQSeqlens;
    sizes.cuMaskRows = sizes.cuQSeqlens;
    sizes.qBuf = params.elementSize * paddedQueryTokens * queryHiddenUnits;
    sizes.kBuf = params.elementSize * paddedKvTokens * kvHiddenUnits;
    sizes.vBuf = sizes.kBuf;
    sizes.qkBuf = params.elementSize * scoreElements;
    sizes.qkvBuf = sizes.qBuf;
    sizes.qkFloatBuf = sizeof(float) * scoreElements;
    sizes.paddingOffset = sizeof(int) * paddedQueryTokens;
    sizes.encoderPaddingOffset = sizeof(int) * paddedKvTokens;
    sizes.tokensInfo = sizeof(int2) * params.packedTokenCount;
    return sizes;
}

} // namespace common::op

TRTLLM_NAMESPACE_END
