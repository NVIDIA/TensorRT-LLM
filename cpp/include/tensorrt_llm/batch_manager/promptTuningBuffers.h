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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/promptTuningParams.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{

class PromptTuningBuffers
{

public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using ITensor = tensorrt_llm::runtime::ITensor;
    using TensorPtr = runtime::ITensor::SharedPtr;

    runtime::PromptTuningParams mPromptTuningParams;
    SizeType32 mMaxPromptVocabSize;

    bool runtimeIsChunkedContext;

    bool mChunkPtableInitialized{false};
    std::optional<std::array<TensorPtr, 2>> mChunkPtableBuffers;
    std::optional<std::vector<std::vector<SizeType32>>> mChunkPtableBufferStartPositions;
    size_t mChunkPtableCurrentIndex{0};

    PromptTuningBuffers(SizeType32 maxBatchSize, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void validate(std::optional<TensorPtr> const& optReqPromptEmbeddingTable,
        std::optional<SizeType32> const& optReqPromptVocabSize);

    void fill(RequestVector const& contextRequests, RequestVector const& genRequests,
        runtime::BufferManager const& manager, bool packed);

    void initializeChunkPtableBuffers(
        runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig, SizeType32 contextChunkSize)
    {
        if (mChunkPtableInitialized)
        {
            return;
        }

        std::array<TensorPtr, 2> buffers;
        std::vector<std::vector<SizeType32>> startPositions(2);

        for (int i = 0; i < 2; i++)
        {
            // Initialize each buffer's positions with 0
            startPositions[i].emplace_back(0);

            buffers[i] = manager.gpu(runtime::ITensor::makeShape({contextChunkSize, modelConfig.getHiddenSize()}),
                nvinfer1::DataType::kHALF); // TODO: change to embedding table data type
        }

        // Assign to optional members
        mChunkPtableBuffers = std::move(buffers);
        mChunkPtableBufferStartPositions = std::move(startPositions);

        // Initialize position
        mChunkPtableCurrentIndex = 0;
        mChunkPtableInitialized = true;
    }

    // GPU ping-pong buffer
    void moveToNextChunkPtableBuffer()
    {
        // Switch ping-pong buffer
        mChunkPtableCurrentIndex = 1 - mChunkPtableCurrentIndex;
        clearBufferStartPositions(mChunkPtableCurrentIndex);
    }

    size_t getChunkPtableCurrentIndex()
    {
        return mChunkPtableCurrentIndex;
    }

    [[nodiscard]] TensorPtr& getChunkPtableBuffer(size_t index)
    {
        if (!mChunkPtableBuffers.has_value())
        {
            TLLM_THROW("Chunk ptable buffers not initialized");
        }
        if (!mChunkPtableBuffers.value()[index])
        {
            TLLM_THROW("Chunk ptable buffer at index %zu is null", index);
        }
        return mChunkPtableBuffers.value()[index];
    }

    [[nodiscard]] SizeType32 getChunkPtableBufferSliceSize(size_t index, size_t batchIdx)
    {
        if (!mChunkPtableBufferStartPositions.has_value())
        {
            return 0;
        }

        // Check if batchIdx is within bounds
        if (batchIdx + 1 >= mChunkPtableBufferStartPositions.value()[index].size())
        {
            TLLM_THROW("Batch index %zu + 1 out of bounds for buffer %zu (size: %zu)", batchIdx, index,
                mChunkPtableBufferStartPositions.value()[index].size());
        }

        // For other batches, return difference from previous position
        return mChunkPtableBufferStartPositions.value()[index][batchIdx + 1]
            - mChunkPtableBufferStartPositions.value()[index][batchIdx];
    }

    [[nodiscard]] SizeType32 getChunkPtableBufferStartPosition(size_t index, size_t batchIdx)
    {
        if (!mChunkPtableBufferStartPositions.has_value())
        {
            return 0;
        }

        // Check if batchIdx is within bounds
        if (batchIdx >= mChunkPtableBufferStartPositions.value()[index].size())
        {
            TLLM_THROW("Batch index %zu out of bounds for buffer %zu (size: %zu)", batchIdx, index,
                mChunkPtableBufferStartPositions.value()[index].size());
        }

        // For first batch, return the value directly
        if (batchIdx == 0)
        {
            return mChunkPtableBufferStartPositions.value()[index][0];
        }

        // For other batches, return difference from previous position
        return mChunkPtableBufferStartPositions.value()[index][batchIdx]
            - mChunkPtableBufferStartPositions.value()[index][batchIdx - 1];
    }

    void updateBufferStartPosition(size_t index, SizeType32 numRows)
    {
        if (!mChunkPtableBufferStartPositions.has_value())
        {
            return;
        }
        // Add new position as sum of previous position plus new tokens
        auto& positions = mChunkPtableBufferStartPositions.value()[index];
        positions.push_back(positions.back() + numRows);
    }

    void clearBufferStartPositions(size_t index)
    {
        if (mChunkPtableBufferStartPositions.has_value())
        {
            mChunkPtableBufferStartPositions.value()[index].clear();
            mChunkPtableBufferStartPositions.value()[index].emplace_back(0);
        }
    }
};

} // namespace tensorrt_llm::batch_manager
