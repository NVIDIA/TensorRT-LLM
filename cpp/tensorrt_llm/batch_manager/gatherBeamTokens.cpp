/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/gatherBeamTokens.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

using SizeType32 = tr::SizeType32;
using TokenIdType = tr::TokenIdType;

void buildGatheredBeamTokensForCallback(
    DecoderInputBuffers& inputBuffers,
    tr::decoder::DecoderState const& decoderState,
    tr::BufferManager const& bufferManager)
{
    NVTX3_SCOPED_RANGE(buildGatheredBeamTokensForCallback);

    auto const numReqs = inputBuffers.decoderRequests.size();
    inputBuffers.gatheredBeamTokensForCallback.assign(numReqs, std::nullopt);

    for (size_t i = 0; i < numReqs; ++i)
    {
        auto const& llmReq = inputBuffers.decoderRequests[i];
        auto const beamWidth = llmReq->getBeamWidthByIter(true);

        // Nothing to gather for single-beam requests.
        if (beamWidth <= 1)
        {
            continue;
        }
        // Skip if no callback — the gather would have no consumer.
        if (!llmReq->mLogitsPostProcessor && !llmReq->mApplyLogitsPostProcessorBatched)
        {
            continue;
        }

        auto const seqSlot = llmReq->mSeqSlot.value();
        auto const& mTokens = llmReq->getTokens(); // slot-accumulated [beamWidth][promptLen + numGen_b]
        auto const promptLen = llmReq->mPromptLen;

        // Per-beam generated counts — beams may have diverging lengths.
        std::vector<SizeType32> numGeneratedPerBeam(beamWidth);
        SizeType32 maxGenerated = 0;
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto const beamGenerated = static_cast<SizeType32>(mTokens[beam].size()) - promptLen;
            numGeneratedPerBeam[beam] = beamGenerated;
            maxGenerated = std::max(maxGenerated, beamGenerated);
        }

        // Step 0/1 cannot have a beam reorder yet.
        if (maxGenerated <= 1)
        {
            continue;
        }

        // GPU → host copy of this slot's parentIds.
        // Shape: [beamWidth, maxSeqLength]
        auto parentIdsDevice = tr::ITensor::at(decoderState.getParentIds(), {seqSlot});
        auto parentIdsHost
            = tr::BufferManager::pinnedPool(parentIdsDevice->getShape(), nvinfer1::DataType::kINT32);
        bufferManager.copy(*parentIdsDevice, *parentIdsHost);
        bufferManager.getStream().synchronize();

        auto const* parentIdsData = tr::bufferCast<TokenIdType>(*parentIdsHost);
        auto const maxSeqLength = static_cast<SizeType32>(parentIdsDevice->getShape().d[1]);

        DecoderInputBuffers::BeamTokens gathered(beamWidth);
        bool anyReorderNeeded = false;

        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto const beamNumGenerated = numGeneratedPerBeam[beam];

            if (beamNumGenerated <= 0)
            {
                gathered[beam] = mTokens[beam];
                continue;
            }

            // Trace parentIds backward to find the source slot at each generation step.
            std::vector<SizeType32> slotAtStep(beamNumGenerated);
            SizeType32 slot = beam;
            for (SizeType32 g = beamNumGenerated - 1; g >= 0; --g)
            {
                slotAtStep[g] = slot;
                if (g > 0)
                {
                    slot = parentIdsData[slot * maxSeqLength + (promptLen + g)];
                }
            }

            // Check if any beam crossing happened.
            for (SizeType32 g = 0; g < beamNumGenerated; ++g)
            {
                if (slotAtStep[g] != beam)
                {
                    anyReorderNeeded = true;
                    break;
                }
            }

            // Materialize coherent path: shared prompt + traced generated tokens.
            auto& tokens = gathered[beam];
            tokens.reserve(static_cast<size_t>(promptLen) + static_cast<size_t>(beamNumGenerated));
            tokens.insert(tokens.end(), mTokens[0].begin(), mTokens[0].begin() + promptLen);
            for (SizeType32 g = 0; g < beamNumGenerated; ++g)
            {
                tokens.push_back(mTokens[slotAtStep[g]][promptLen + g]);
            }
        }

        if (anyReorderNeeded)
        {
            inputBuffers.gatheredBeamTokensForCallback[i] = std::move(gathered);
        }
    }
}

} // namespace tensorrt_llm::batch_manager
