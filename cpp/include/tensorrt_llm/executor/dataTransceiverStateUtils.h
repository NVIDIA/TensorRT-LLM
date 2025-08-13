/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serialization.h"
#include <vector>

namespace tensorrt_llm::executor
{

// Utility functions for creating and serializing DataTransceiverState

/**
 * @brief Create a serialized DataTransceiverState with socket communication state
 *
 * @param nbKvHeadsPerLayer Vector of number of KV heads per layer
 * @param sizePerHead Size of each attention head
 * @param tokensPerBlock Number of tokens per block
 * @param tensorParallelism Tensor parallelism size
 * @param pipelineParallelism Pipeline parallelism size
 * @param dataType Data type for the cache
 * @param socketAddresses Vector of socket addresses for communication
 * @param attentionType Attention type (DEFAULT or MLA)
 * @param kvFactor KV factor (default: 2)
 * @param enableAttentionDP Whether to enable attention data parallelism
 * @param dpRank Data parallelism rank (default: 0)
 * @param dpSize Data parallelism size (default: 0)
 * @param rank Current rank
 * @return std::vector<char> The serialized DataTransceiverState as bytes
 */
inline std::vector<char> createDataTransceiverStateSocket(std::vector<SizeType32> const& nbKvHeadsPerLayer,
    SizeType32 sizePerHead, SizeType32 tokensPerBlock, SizeType32 tensorParallelism, SizeType32 pipelineParallelism,
    nvinfer1::DataType dataType, std::vector<std::string> const& socketAddresses,
    kv_cache::CacheState::AttentionType attentionType, int kvFactor, bool enableAttentionDP, int dpRank, int dpSize,
    int rank)
{
    // Create CacheState using the simpler constructor
    kv_cache::CacheState cacheState(nbKvHeadsPerLayer, sizePerHead, tokensPerBlock, tensorParallelism,
        pipelineParallelism, dataType, attentionType, kvFactor, enableAttentionDP, dpRank, dpSize);

    // Create Socket CommState
    std::vector<kv_cache::SocketState> socketStates;
    for (size_t i = 0; i < socketAddresses.size(); ++i)
    {
        kv_cache::SocketState socketState{static_cast<uint16_t>(8000 + i), socketAddresses[i]};
        socketStates.emplace_back(std::move(socketState));
    }

    kv_cache::CommState commState(std::move(socketStates), rank);

    // Create DataTransceiverState
    DataTransceiverState state(std::move(cacheState), std::move(commState));

    // Serialize and return the serialized data
    return Serialization::serialize(state);
}

/**
 * @brief Create a serialized DataTransceiverState with agent communication state
 *
 * @param nbKvHeadsPerLayer Vector of number of KV heads per layer
 * @param sizePerHead Size of each attention head
 * @param tokensPerBlock Number of tokens per block
 * @param tensorParallelism Tensor parallelism size
 * @param pipelineParallelism Pipeline parallelism size
 * @param dataType Data type for the cache
 * @param agentNames Vector of agent names for communication
 * @param attentionType Attention type (DEFAULT or MLA)
 * @param kvFactor KV factor (default: 2)
 * @param enableAttentionDP Whether to enable attention data parallelism
 * @param dpRank Data parallelism rank (default: 0)
 * @param dpSize Data parallelism size (default: 0)
 * @param rank Current rank
 * @return std::vector<char> The serialized DataTransceiverState as bytes
 */
inline std::vector<char> createDataTransceiverStateAgent(std::vector<SizeType32> const& nbKvHeadsPerLayer,
    SizeType32 sizePerHead, SizeType32 tokensPerBlock, SizeType32 tensorParallelism, SizeType32 pipelineParallelism,
    nvinfer1::DataType dataType, std::vector<std::string> const& agentNames,
    kv_cache::CacheState::AttentionType attentionType, int kvFactor, bool enableAttentionDP, int dpRank, int dpSize,
    int rank)
{
    // Create CacheState using the simpler constructor
    kv_cache::CacheState cacheState(nbKvHeadsPerLayer, sizePerHead, tokensPerBlock, tensorParallelism,
        pipelineParallelism, dataType, attentionType, kvFactor, enableAttentionDP, dpRank, dpSize);

    // Create Agent CommState
    std::vector<kv_cache::AgentState> agentStates;
    for (size_t i = 0; i < agentNames.size(); ++i)
    {
        agentStates.emplace_back(agentNames[i], "127.0.0.1:" + std::to_string(8000 + i));
    }

    kv_cache::CommState commState(std::move(agentStates), rank);

    // Create DataTransceiverState
    DataTransceiverState state(std::move(cacheState), std::move(commState));

    // Serialize and return the serialized data
    return Serialization::serialize(state);
}

} // namespace tensorrt_llm::executor
