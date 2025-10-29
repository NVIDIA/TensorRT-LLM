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
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager
{

using BufferManager = tensorrt_llm::runtime::BufferManager;
using TensorPtr = runtime::ITensor::SharedPtr;
using ITensor = runtime::ITensor;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

runtime::CudaEvent UpdateDecoderBuffers::operator()(runtime::ModelConfig const& modelConfig,
    DecoderOutputBuffers& decoderOutputBuffers, runtime::BufferManager const& copyBufferManager,
    runtime::decoder::DecoderState const& decoderState, bool returnLogProbs,
    runtime::CudaEvent const& decoderFinishEvent) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(updateDecoderBuffers);

    // Chain copy after decoder event, using a different stream
    copyBufferManager.getStream().wait(decoderFinishEvent);

    copyBufferManager.copy(*decoderState.getAllNewTokens(), *decoderOutputBuffers.newOutputTokensHost);
    copyBufferManager.copy(*decoderState.getSequenceLengths(), *decoderOutputBuffers.sequenceLengthsHost);

    auto const finishedSumDevice = decoderState.getFinishedSum();
    copyBufferManager.copy(*finishedSumDevice, *decoderOutputBuffers.finishedSumHost);
    auto const finishReasonsDevice = decoderState.getFinishReasons();
    copyBufferManager.copy(*finishReasonsDevice, *decoderOutputBuffers.finishReasonsHost);

    if (returnLogProbs)
    {
        copyBufferManager.copy(*decoderState.getCumLogProbs(), *decoderOutputBuffers.cumLogProbsHost);
        copyBufferManager.copy(*decoderState.getLogProbs(), *decoderOutputBuffers.logProbsHost);
    }

    if (modelConfig.getSpeculativeDecodingMode().predictsDraftTokens())
    {
        // TODO: keep data on device for next iteration
        copyBufferManager.copy(*decoderState.getNextDraftTokens(), *decoderOutputBuffers.nextDraftTokensHost);

        if (modelConfig.getSpeculativeDecodingMode().variableDraftLength())
        {
            copyBufferManager.copy(
                *decoderState.getNextDraftTokensLengths(), *decoderOutputBuffers.nextDraftTokensLengthsHost);
            copyBufferManager.copy(
                *decoderState.getPrevDraftTokensLengths(), *decoderOutputBuffers.prevDraftTokensLengthsHost);
        }
    }

    runtime::CudaEvent copyEvent{};
    copyBufferManager.getStream().record(copyEvent);
    // Store the event for later sync. Sync stream before calling next decoder. Sync host before updating requests.
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return copyEvent;
}

} // namespace tensorrt_llm::batch_manager
