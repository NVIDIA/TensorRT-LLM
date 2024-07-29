/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/generationConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

using namespace tensorrt_llm::runtime;

GenerationConfig GenerationConfig::fromInput(ITensor const& inputIds, ITensor& inputLengthsHost, bool const inputPacked,
    SizeType32 const beamWidth, SizeType32 const maxAttentionWindow, SizeType32 const sinkTokenLength,
    SizeType32 const maxSequenceLength)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = static_cast<SizeType32>(inputLengthsHost.getSize());

    auto inputLengthsHostBuffer = BufferRange<SizeType32>(inputLengthsHost);
    SizeType32 maxInputLength
        = *std::max_element(inputLengthsHostBuffer.begin(), inputLengthsHostBuffer.begin() + batchSize);

    auto const& inputShape = inputIds.getShape();
    SizeType32 inputLengthSum{0};
    if (inputPacked)
    {
        inputLengthSum = std::accumulate(inputLengthsHostBuffer.begin(), inputLengthsHostBuffer.begin() + batchSize, 0);
        TLLM_CHECK_WITH_INFO(inputShape.nbDims == 1 || inputShape.nbDims == 2,
            "Packed input must have shape [<sum of input lengths>] or [1, <sum of input lengths>].");
        if (inputShape.nbDims == 1)
        {
            TLLM_CHECK_WITH_INFO(inputShape.d[0] == inputLengthSum,
                "Packed 1D input must have shape [<sum of input lengths>]. Expected (Infer from inputLengths): [%d], "
                "supplied: [" FMT_DIM "]",
                inputLengthSum, inputShape.d[0]);
        }
        else if (inputShape.nbDims == 2)
        {
            TLLM_CHECK_WITH_INFO(inputShape.d[1] == inputLengthSum,
                "Packed 2D input must have shape [1, <sum of input lengths>]. Expected (Infer from inputLengths): [1, "
                "%d], supplied: [" FMT_DIM ", " FMT_DIM "]",
                inputLengthSum, inputShape.d[0], inputShape.d[1]);
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(inputShape.d[0] == batchSize && inputShape.d[1] >= maxInputLength,
            "Padded input must have shape [batch size, max input length]");
        maxInputLength = inputShape.d[1];
    }

    TLLM_CHECK_WITH_INFO(maxInputLength < maxSequenceLength,
        "Max input length is equal to or larger that maxSequenceLength given in setup. No new tokens can be "
        "generated.");

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return GenerationConfig{
        batchSize, beamWidth, maxInputLength, maxAttentionWindow, sinkTokenLength, maxSequenceLength, inputLengthSum};
}
