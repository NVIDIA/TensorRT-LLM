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

#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{
using SizeType32 = MakeDecodingBatchInputOutput::SizeType32;
using TensorPtr = MakeDecodingBatchInputOutput::TensorPtr;

namespace
{
std::vector<bool> computeActiveVec(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 maxNumSequences)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::vector<bool> active(maxNumSequences, false);
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const seqSlot = llmReq->mSeqSlot.value();
            if (llmReq->isGenerationInProgressState() || llmReq->isLastContextChunk())
            {
                active[seqSlot] = true;
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return active;
}

void copySequenceLengths(RequestVector const& contextRequests, RequestVector const& generationRequests,
    DecoderInputBuffers const& inputBuffers, TensorPtr const& sequenceLengths, SizeType32 beamWidth,
    runtime::BufferManager const& manager, runtime::CudaStream const& stream)
{
    auto const batchSize = contextRequests.size() + generationRequests.size();
    auto batchSlotsView = tr::ITensor::slice(inputBuffers.forwardBatchSlotsRequestOrder, 0, batchSize);
    auto fillValuesView = tr::ITensor::slice(inputBuffers.fillValues, 0, batchSize);

    auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlotsView);
    auto fillValuesRange = tr::BufferRange<SizeType32>(*fillValuesView);

    // fill buffers on host
    SizeType32 batchIdx{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const currentSequenceLen = llmReq->mPromptLen + llmReq->getMaxNumGeneratedTokens();
            // Get position of the current sequence in the decoder
            auto const seqSlot = llmReq->mSeqSlot.value();
            batchSlotsRange[batchIdx] = seqSlot;
            fillValuesRange[batchIdx] = currentSequenceLen;
            ++batchIdx;
        }
    }

    // copy sequence lengths
    {
        auto batchSlotsDeviceView = tr::ITensor::slice(inputBuffers.forwardBatchSlotsRequestOrderDevice, 0, batchSize);
        auto fillValuesViewDevice = tr::ITensor::slice(inputBuffers.fillValuesDevice, 0, batchSize);

        manager.copy(*batchSlotsView, *batchSlotsDeviceView);
        manager.copy(*fillValuesView, *fillValuesViewDevice);
        tr::kernels::invokeFillBatch(*sequenceLengths, *batchSlotsDeviceView, beamWidth, *fillValuesViewDevice, stream);
    }
}
} // namespace

std::tuple<std::unique_ptr<tr::decoder_batch::Input>, std::unique_ptr<tr::decoder_batch::Output>>
MakeDecodingBatchInputOutput::operator()(RequestVector const& contextRequests, RequestVector const& generationRequests,
    DecoderBuffers& decoderBuffers, DecoderInputBuffers const& inputBuffers,
    runtime::decoder::DecoderState& decoderState, runtime::ModelConfig const& modelConfig, SizeType32 maxNumSequences,
    SizeType32 beamWidth, runtime::BufferManager const& manager, runtime::CudaStream const& stream,
    OptionalRef<RuntimeBuffers> fusedRuntimeBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const active = computeActiveVec(contextRequests, generationRequests, maxNumSequences);
    auto decodingInput = std::make_unique<tr::decoder_batch::Input>(decoderBuffers.logits, active);
    decodingInput->batchSlots = inputBuffers.forwardBatchSlots;

    decodingInput->cacheIndirection = decoderBuffers.cacheIndirectionInput;

    if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
    {
        decodingInput->predictedDraftLogits = decoderBuffers.draftBuffers.predictedDraftLogits;
    }

    if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        TLLM_CHECK(fusedRuntimeBuffers);
        // requires mCtxGenFusion == true
        decodingInput->batchSlotsRequestOrder = fusedRuntimeBuffers->seqSlots;
        decodingInput->explicitDraftTokensInputs = fusedRuntimeBuffers->explicitDraftTokensBuffers->engineOutputs;
        decodingInput->explicitDraftTokensLastInputs = fusedRuntimeBuffers->explicitDraftTokensBuffers->engineInputs;
    }
    else if (modelConfig.getSpeculativeDecodingMode().isEagle())
    {
        TLLM_CHECK(fusedRuntimeBuffers);
        // requires mCtxGenFusion == true
        decodingInput->batchSlotsRequestOrder = fusedRuntimeBuffers->seqSlots;
        decodingInput->eagleInputs = fusedRuntimeBuffers->eagleBuffers->engineOutputs;
        decodingInput->eagleLastInputs = fusedRuntimeBuffers->eagleBuffers->engineInputs;
    }

    copySequenceLengths(contextRequests, generationRequests, inputBuffers,
        decoderState.getJointDecodingOutput().lengths, beamWidth, manager, stream);

    auto decodingOutput = std::make_unique<tr::decoder_batch::Output>();
    decodingOutput->cacheIndirection = decoderBuffers.cacheIndirectionOutput;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(decodingInput), std::move(decodingOutput)};
}

} // namespace tensorrt_llm::batch_manager
