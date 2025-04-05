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

#include "tensorrt_llm/batch_manager/updateDecoderBuffers.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager
{

using BufferManager = tensorrt_llm::runtime::BufferManager;
using TensorPtr = runtime::ITensor::SharedPtr;
using ITensor = runtime::ITensor;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

runtime::CudaEvent UpdateDecoderBuffers::operator()(runtime::ModelConfig const& modelConfig,
    DecoderBuffers& decoderBuffers, runtime::BufferManager const& copyBufferManager,
    runtime::GptDecoderBatched const& decoder, bool returnLogProbs, runtime::CudaEvent const& decoderFinishEvent) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(updateDecoderBuffers);

    // Chain copy after decoder event, using a different stream
    copyBufferManager.getStream().wait(decoderFinishEvent);

    decoderBuffers.newOutputTokens = decoder.getDecoderState().getAllNewTokens();

    copyBufferManager.copy(*decoderBuffers.newOutputTokens, *decoderBuffers.newOutputTokensHost);
    copyBufferManager.copy(*decoderBuffers.sequenceLengths, *decoderBuffers.sequenceLengthsHost);

    auto const finishedSumDevice = decoder.getDecoderState().getFinishedSum();
    copyBufferManager.copy(*finishedSumDevice, *decoderBuffers.finishedSumHost);
    auto const finishReasonsDevice = decoder.getDecoderState().getFinishReasons();
    copyBufferManager.copy(*finishReasonsDevice, *decoderBuffers.finishReasonsHost);

    if (returnLogProbs)
    {
        decoderBuffers.cumLogProbs = decoder.getDecoderState().getCumLogProbs();
        decoderBuffers.logProbs = decoder.getDecoderState().getLogProbs();
        copyBufferManager.copy(*decoderBuffers.cumLogProbs, *decoderBuffers.cumLogProbsHost);
        copyBufferManager.copy(*decoderBuffers.logProbs, *decoderBuffers.logProbsHost);
    }

    if (modelConfig.getSpeculativeDecodingMode().predictsDraftTokens())
    {
        // TODO(rkobus): keep data on device for next iteration
        decoderBuffers.draftBuffers.nextDraftTokensDevice = decoder.getDecoderState().getNextDraftTokens();
        copyBufferManager.copy(
            *decoderBuffers.draftBuffers.nextDraftTokensDevice, *decoderBuffers.draftBuffers.nextDraftTokensHost);

        if (modelConfig.getSpeculativeDecodingMode().variableDraftLength())
        {
            decoderBuffers.draftBuffers.nextDraftTokensLengthsDevice
                = decoder.getDecoderState().getNextDraftTokensLengths();
            decoderBuffers.draftBuffers.prevDraftTokensLengthsDevice
                = decoder.getDecoderState().getPrevDraftTokensLengths();
            copyBufferManager.copy(*decoderBuffers.draftBuffers.nextDraftTokensLengthsDevice,
                *decoderBuffers.draftBuffers.nextDraftTokensLengthsHost);
            copyBufferManager.copy(*decoderBuffers.draftBuffers.prevDraftTokensLengthsDevice,
                *decoderBuffers.draftBuffers.prevDraftTokensLengthsHost);
        }
    }

    if (modelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
    {
        decoderBuffers.draftBuffers.acceptedLengthsCumSumDevice = decoder.getDecoderState().getAcceptedLengthsCumSum();
        decoderBuffers.draftBuffers.acceptedPackedPathsDevice = decoder.getDecoderState().getAcceptedPackedPaths();
    }

    runtime::CudaEvent copyEvent{};
    copyBufferManager.getStream().record(copyEvent);
    // Store the event for later sync. Sync stream before calling next decoder. Sync host before updating requests.
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return copyEvent;
}

} // namespace tensorrt_llm::batch_manager
