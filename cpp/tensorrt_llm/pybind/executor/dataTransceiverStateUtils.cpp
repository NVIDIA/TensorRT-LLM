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

#include "dataTransceiverStateUtils.h"
#include "tensorrt_llm/executor/dataTransceiverStateUtils.h"
#include "tensorrt_llm/executor/types.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Use the types from the executor namespace
using tensorrt_llm::executor::SizeType32;
using namespace tensorrt_llm::executor;

namespace tensorrt_llm::pybind::executor
{

void bindDataTransceiverStateUtils(py::module_& m)
{
    auto dataTransceiverUtils = m.def_submodule("data_transceiver_utils", "DataTransceiverState utility functions");

    dataTransceiverUtils.def(
        "create_data_transceiver_state_socket",
        [](std::vector<SizeType32> const& nbKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
            SizeType32 tensorParallelism, SizeType32 pipelineParallelism, nvinfer1::DataType dataType,
            std::vector<std::string> const& socketAddresses, kv_cache::CacheState::AttentionType attentionType,
            int kvFactor, bool enableAttentionDP, int dpRank, int dpSize, int rank)
        {
            return createDataTransceiverStateSocket(nbKvHeadsPerLayer, sizePerHead, tokensPerBlock, tensorParallelism,
                pipelineParallelism, dataType, socketAddresses, attentionType, kvFactor, enableAttentionDP, dpRank,
                dpSize, rank);
        },
        py::arg("nb_kv_heads_per_layer"), py::arg("size_per_head"), py::arg("tokens_per_block"),
        py::arg("tensor_parallelism"), py::arg("pipeline_parallelism"), py::arg("data_type"),
        py::arg("socket_addresses"), py::arg("attention_type") = kv_cache::CacheState::AttentionType::kDEFAULT,
        py::arg("kv_factor") = 2, py::arg("enable_attention_dp") = false, py::arg("dp_rank") = 0,
        py::arg("dp_size") = 0, py::arg("rank") = 0, "Create a DataTransceiverState with socket communication state");

    dataTransceiverUtils.def(
        "create_data_transceiver_state_agent",
        [](std::vector<SizeType32> const& nbKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
            SizeType32 tensorParallelism, SizeType32 pipelineParallelism, nvinfer1::DataType dataType,
            std::vector<std::string> const& agentNames, kv_cache::CacheState::AttentionType attentionType, int kvFactor,
            bool enableAttentionDP, int dpRank, int dpSize, int rank)
        {
            return createDataTransceiverStateAgent(nbKvHeadsPerLayer, sizePerHead, tokensPerBlock, tensorParallelism,
                pipelineParallelism, dataType, agentNames, attentionType, kvFactor, enableAttentionDP, dpRank, dpSize,
                rank);
        },
        py::arg("nb_kv_heads_per_layer"), py::arg("size_per_head"), py::arg("tokens_per_block"),
        py::arg("tensor_parallelism"), py::arg("pipeline_parallelism"), py::arg("data_type"), py::arg("agent_names"),
        py::arg("attention_type") = kv_cache::CacheState::AttentionType::kDEFAULT, py::arg("kv_factor") = 2,
        py::arg("enable_attention_dp") = false, py::arg("dp_rank") = 0, py::arg("dp_size") = 0, py::arg("rank") = 0,
        "Create a DataTransceiverState with agent communication state");
}

} // namespace tensorrt_llm::pybind::executor
