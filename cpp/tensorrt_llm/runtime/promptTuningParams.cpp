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

#include "tensorrt_llm/runtime/promptTuningParams.h"

namespace tensorrt_llm::runtime
{

void PromptTuningParams::fillTasksTensor(TensorPtr tasksHost, const SizeType batchSize,
    const SizeType numContextRequests, const std::vector<SizeType>& reqBeamWidths,
    const std::vector<SizeType>& reqPromptLengths, BufferManager const& manager, bool packedInput)
{
    auto const& tasksHostShape = tasksHost->getShape();
    TLLM_CHECK_WITH_INFO(tasksHostShape.nbDims == 1, "tasksHost expected to have dimension [batchSize]");
    TLLM_CHECK_WITH_INFO(tasksHostShape.d[0] == batchSize, "tasksHost expected to have dimension [batchSize]");

    auto const tasksHostPtr = bufferCast<SizeType const>(*tasksHost);

    bool validInput = packedInput || numContextRequests == batchSize || numContextRequests == 0;
    TLLM_CHECK_WITH_INFO(validInput,
        "fillTasksTensor function with packed inputs must be called with only context requests or only generation "
        "requests.");

    bool validShapes = (static_cast<SizeType>(reqBeamWidths.size()) == batchSize
        && static_cast<SizeType>(reqPromptLengths.size()) == numContextRequests
        && static_cast<SizeType>(promptTuningEnabled.size()) == batchSize);
    TLLM_CHECK_WITH_INFO(validShapes,
        "Invalid inputs to fillTasksTensor function. reqBeamWidths and reqPtuningEnabled size must be batchSize and "
        "propmtLenghts size must be numContextRequests");

    SizeType totalInputSize = 0;
    std::vector<SizeType> promptTasksHost;
    for (SizeType bid = 0; bid < batchSize; bid++)
    {
        SizeType taskId = promptTuningEnabled[bid] ? tasksHostPtr[bid] : 0;
        if (packedInput)
        {
            if (bid < numContextRequests)
            {
                totalInputSize += reqPromptLengths[bid];
                promptTasksHost.insert(promptTasksHost.end(), reqPromptLengths[bid], taskId);
            }
            else
            {
                for (SizeType beam = 0; beam < reqBeamWidths[bid]; ++beam)
                {
                    promptTasksHost.insert(promptTasksHost.end(), 1, taskId);
                    totalInputSize++;
                }
            }
        }
        else
        {
            if (bid < numContextRequests)
            {
                promptTasksHost.push_back(taskId);
                ++totalInputSize;
            }
            else
            {
                promptTasksHost.insert(promptTasksHost.end(), reqBeamWidths[bid], taskId);
                totalInputSize += reqBeamWidths[bid];
            }
        }
    }

    if (packedInput)
    {
        tasks = manager.copyFrom(
            promptTasksHost, runtime::ITensor::makeShape({totalInputSize}), runtime::MemoryType::kGPU);
    }
    else
    {
        tasks = manager.copyFrom(
            promptTasksHost, runtime::ITensor::makeShape({totalInputSize, 1}), runtime::MemoryType::kGPU);
    }
}

} // namespace tensorrt_llm::runtime
