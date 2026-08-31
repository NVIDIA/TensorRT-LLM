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

#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/runtime/bufferManager.h"

namespace tensorrt_llm::batch_manager
{

void MedusaBuffers::reshape(SizeType32 /* numCtxSequences */, SizeType32 numGenSequences, SizeType32 tokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto attentionPackedMaskShape = attentionPackedMaskDevice->getShape();
    attentionPackedMaskShape.d[0] = numGenSequences * tokensPerStep;
    attentionPackedMaskDevice->reshape(attentionPackedMaskShape);

    auto medusaGenerationLengthsShape = medusaGenerationLengthsDevice->getShape();
    medusaGenerationLengthsShape.d[0] = numGenSequences;
    medusaGenerationLengthsDevice->reshape(medusaGenerationLengthsShape);

    auto medusaPositionOffsetsShape = medusaPositionOffsetsDevice->getShape();
    medusaPositionOffsetsShape.d[0] = numGenSequences;
    medusaPositionOffsetsDevice->reshape(medusaPositionOffsetsShape);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void MedusaBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", attentionPackedMaskDevice);
    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", medusaGenerationLengthsDevice);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", medusaPositionOffsetsDevice);
    inputBuffers.insert_or_assign("spec_decoding_use", medusaUseSpecDecoding);
    if (worldConfig.isLastPipelineParallelRank())
    {
        outputBuffers.insert_or_assign("medusa_logits", medusaLogitsDevice);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
