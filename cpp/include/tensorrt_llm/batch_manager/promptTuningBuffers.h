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

    PromptTuningBuffers(SizeType32 maxBatchSize, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    PromptTuningBuffers(SizeType32 maxBatchSize, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, bool promptTableOffloading);

    void validate(std::optional<TensorPtr> const& optReqPromptEmbeddingTable,
        std::optional<SizeType32> const& optReqPromptVocabSize);

    void fill(RequestVector const& contextRequests, RequestVector const& genRequests,
        runtime::BufferManager const& manager, bool packed);

    /*
     * The below functions are specific for Chunked Prefill mode
     * Chunk Ptable with Ping-Pong Buffer Implementation
     * -----------------------------------------------
     *
     * Overview:
     * The chunk ptable (prompt tuning table) system uses a ping-pong buffer mechanism to efficiently
     * manage large embedding tables when operating in context Prefill mode. This allows
     * for processing of large embedding tables by loading them in chunks from CPU to GPU memory,
     * enabling support for tables that exceed available GPU memory.
     *
     * Key Components:
     * 1. Ping-Pong Buffers (mChunkPtableBuffers):
     *    - Two alternating GPU buffers that store chunks of the embedding table
     *    - While the current buffer is being processed by the model,
     *      the next chunk can be asynchronously loaded into the other buffer
     *    - Managed through mChunkPtableCurrentIndex (toggles between 0 and 1)
     * 2. Start Positions Tracking (mChunkPtableBufferStartPositions):
     *    - Mainly used for multi-batch processing
     *    - Maintains the starting position of each batch's data within each buffer
     *    - Maintained separately for each ping-pong buffer
     *
     * Memory Optimization:
     * - Only two GPU buffers are maintained regardless of total embedding table size
     * - Each buffer size is limited to contextChunkSize * hiddenSize
     * - Efficient memory usage through chunk-based processing
     */

    bool mPromptTableOffloading;

    bool mChunkPtableInitialized{false};
    std::optional<std::array<TensorPtr, 2>> mChunkPtableBuffers;
    std::optional<std::vector<std::vector<SizeType32>>> mChunkPtableBufferStartPositions;
    size_t mChunkPtableCurrentIndex{0};

    void initializeChunkPtableBuffers(runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
        SizeType32 contextChunkSize, std::shared_ptr<LlmRequest> const& llmReq);

    void switchChunkPtableBuffer();

    size_t getChunkPtableCurrentIndex();

    [[nodiscard]] TensorPtr& getChunkPtableBuffer(size_t index);

    [[nodiscard]] SizeType32 getChunkPtableBufferSliceSize(size_t index, size_t batchIdx);

    [[nodiscard]] SizeType32 getChunkPtableBufferStartPosition(size_t index, size_t batchIdx);

    void updateBufferStartPosition(size_t index, SizeType32 numRows);

    void clearBufferStartPositions(size_t index);
};

} // namespace tensorrt_llm::batch_manager
