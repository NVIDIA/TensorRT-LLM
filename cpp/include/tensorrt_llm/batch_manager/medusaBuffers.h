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

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/promptTuningParams.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{

class MedusaBuffers
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using ITensor = tensorrt_llm::runtime::ITensor;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    MedusaBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime);

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 tokensPerStep);

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

public:
    TensorPtr medusaLogitsDevice; // [maxAcceptedDraftTokens, maxBatchSize, maxDraftTokens + 1, vocabSizePadded], on gpu

    TensorPtr attentionPackedMaskDevice;     // [maxBatchSize, maxDraftTokens + 1, numPackedMasks], on gpu
    TensorPtr attentionPackedMaskHost;       // [maxBatchSize, maxDraftTokens + 1, numPackedMasks], on pinned

    TensorPtr medusaGenerationLengthsDevice; // [maxBatchSize], on gpu
    TensorPtr medusaGenerationLengthsHost;   // [maxBatchSize], on pinned

    TensorPtr medusaPositionOffsetsDevice;   // [maxBatchSize, maxDraftTokens + 1], on gpu
    TensorPtr medusaPositionOffsetsHost;     // [maxBatchSize, maxDraftTokens + 1], on pinned

    TensorPtr medusaTreeIdsDevice;           // [maxBatchSize, maxDraftTokens + 1], on gpu
    TensorPtr medusaTreeIdsHost;             // [maxBatchSize, maxDraftTokens + 1], on pinned

    TensorPtr medusaPathsDevice;             // [maxBatchSize, maxDraftTokens + 1, maxAcceptedDraftTokens + 1], on gpu
    TensorPtr medusaPathsHost;       // [maxBatchSize, maxDraftTokens + 1, maxAcceptedDraftTokens + 1], on pinned

    TensorPtr medusaUseSpecDecoding; // [1], on cpu

    std::vector<SizeType32> mTopKs;  // [maxAcceptedDraftTokens]
};

} // namespace tensorrt_llm::batch_manager
