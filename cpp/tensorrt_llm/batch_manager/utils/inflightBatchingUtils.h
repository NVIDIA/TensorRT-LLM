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
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager::utils
{
using SizeType32 = runtime::SizeType32;
using TensorPtr = runtime::ITensor::SharedPtr;

template <typename T>
using OptionalRef = common::OptionalRef<T>;

TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector const& generationRequests);

//! @brief Sort requests for functional correctness and performance.
//! @details Sort context requests for moveFinishedContextRequestsToGeneration.
//!          Sort requests by lora task id for performance.
//! @param contextRequests The context requests.
//! @param generationRequests The generation requests.
//! @param chunksPresent Whether context chunks are present.
void sortRequests(RequestVector& contextRequests, RequestVector& generationRequests, bool chunksPresent);

//! @brief Move finished context requests to generation requests.
//! @details This function assumes that the context requests are sorted so that requests with isLastContextChunk() are
//!          at the end of the context requests vector. These requests are moved to the beginning of the generation
//!          requests vector. This means that the order of the requests in context+generation requests is not changed.
//! @param scheduledRequests The scheduled context and generation requests.
void moveFinishedContextRequestsToGeneration(ScheduledRequests& scheduledRequests);

//! @param beforeDecoder    Whether the function is called before the decoder. If it is true, correct the output offset.
//! @param numDroppedTokens The number of dropped tokens for each beam (e.g. when the requests finished early).
//!                         Generation logits for dropped tokens are ignored.
void copyGenerationLogits(RuntimeBuffers::GenerationLogitsCache& generationLogitsCache,
    runtime::BufferManager const& bufferManager, LlmRequest& llmReq, bool beforeDecoder,
    std::vector<SizeType32> const& numDroppedTokens = {});

void copyAdditionalOutputs(std::vector<executor::AdditionalModelOutput> const& additionalModelOutputs,
    RequestVector const& contextRequests, RequestVector const& generationRequests,
    RuntimeBuffers::TensorMap const& outputMap, runtime::BufferManager const& manager);

void terminateRequest(SequenceSlotManager& seqSlotManager, LlmRequest& llmRequest, SizeType32 maxInputLen,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager = std::nullopt,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager = std::nullopt,
    OptionalRef<BasePeftCacheManager> peftCacheManager = std::nullopt, bool pause = false);

std::vector<SizeType32> getRequestBeamWidths(
    RequestVector const& contextRequests, RequestVector const& generationRequests);

class CudaGraphExecutor
{
public:
    CudaGraphExecutor() = default;

    ~CudaGraphExecutor()
    {
        try
        {
            clear();
        }
        catch (std::exception& e)
        {
            TLLM_LOG_EXCEPTION(e);
        }
    }

    bool hasInstance() const
    {
        return mInstance != nullptr;
    }

    void clear();
    void prepareNextGraph(std::unique_ptr<runtime::TllmRuntime>& runtime, SizeType32 nextContextId);
    void launch(runtime::CudaStream const& stream);

private:
    void create(cudaGraph_t const& graph);
    bool update(cudaGraph_t const& graph);
    void uploadToStream(runtime::CudaStream const& stream);

    cudaGraphExec_t mInstance;
};

class CudaGraphExecutorCache
{
    /// @brief LRU cache to store cuda graph instances.
public:
    explicit CudaGraphExecutorCache(runtime::SizeType32 capacity)
        : mCapacity(capacity)
    {
    }

    std::optional<std::shared_ptr<CudaGraphExecutor>> get(BatchState const& state);

    void put(BatchState const& state, std::shared_ptr<CudaGraphExecutor> const& value);

private:
    using BatchStateGraphExecutorPair = std::pair<BatchState, std::shared_ptr<CudaGraphExecutor>>;
    using GraphExecutorLruCache = std::list<BatchStateGraphExecutorPair>;
    SizeType32 mCapacity;
    GraphExecutorLruCache mCache;
    std::unordered_map<BatchState, GraphExecutorLruCache::iterator, BatchStateHash> mMap;
};
} // namespace tensorrt_llm::batch_manager::utils
