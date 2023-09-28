/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/gptJsonConfig.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace
{
using Json = typename nlohmann::json::basic_json;

template <typename FieldType>
FieldType parseJsonFieldOr(Json const& json, std::string_view name, FieldType defaultValue)
{
    auto value = defaultValue;
    try
    {
        value = json.at(name).template get<FieldType>();
    }
    catch (nlohmann::json::out_of_range&)
    {
        // std::cerr << e.what() << '\n';
    }
    return value;
}

template <typename InputType>
GptJsonConfig parseJson(InputType&& i)
{
    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto json = nlohmann::json::parse(i, nullptr, allowExceptions, ingoreComments);

    auto const& builderConfig = json.at("builder_config");
    auto const name = builderConfig.at("name").template get<std::string>();
    auto const precision = builderConfig.at("precision").template get<std::string>();
    auto const tensorParallelism = builderConfig.at("tensor_parallel").template get<SizeType>();
    auto const pipelineParallelism = parseJsonFieldOr(builderConfig, "pipeline_parallel", 1);
    auto const numHeads = builderConfig.at("num_heads").template get<SizeType>() / tensorParallelism;
    auto const hiddenSize = builderConfig.at("hidden_size").template get<SizeType>() / tensorParallelism;
    auto const vocabSize = builderConfig.at("vocab_size").template get<SizeType>();
    auto const numLayers = builderConfig.at("num_layers").template get<SizeType>();

    auto dataType = nvinfer1::DataType::kFLOAT;
    if (!precision.compare("float32"))
        dataType = nvinfer1::DataType::kFLOAT;
    else if (!precision.compare("float16"))
        dataType = nvinfer1::DataType::kHALF;
    else if (!precision.compare("bfloat16"))
        dataType = nvinfer1::DataType::kBF16;
    else
        TLLM_CHECK_WITH_INFO(false, tc::fmtstr("Model data type '%s' not supported", precision.c_str()));

    auto const pagedKvCache = parseJsonFieldOr(builderConfig, "paged_kv_cache", false);
    auto const tokensPerBlock = parseJsonFieldOr(builderConfig, "tokens_per_block", 0);
    auto const quantMode = tc::QuantMode(parseJsonFieldOr(builderConfig, "quant_mode", tc::QuantMode::none().value()));
    auto const numKvHeads
        = parseJsonFieldOr(builderConfig, "num_kv_heads", numHeads * tensorParallelism) / tensorParallelism;
    auto const maxBatchSize = parseJsonFieldOr(builderConfig, "max_batch_size", 0);
    auto const maxInputLen = parseJsonFieldOr(builderConfig, "max_input_len", 0);
    auto const maxOutputLen = parseJsonFieldOr(builderConfig, "max_output_len", 0);

    auto const& pluginConfig = json.at("plugin_config");
    auto const& gptAttentionPlugin = pluginConfig.at("gpt_attention_plugin");
    auto const useGptAttentionPlugin = !gptAttentionPlugin.is_boolean() || gptAttentionPlugin.template get<bool>();
    auto const removeInputPadding = pluginConfig.at("remove_input_padding").template get<bool>();

    auto modelConfig = GptModelConfig{vocabSize, numLayers, numHeads, hiddenSize, dataType};
    modelConfig.useGptAttentionPlugin(useGptAttentionPlugin);
    modelConfig.usePackedInput(removeInputPadding);
    modelConfig.usePagedKvCache(pagedKvCache);
    modelConfig.setTokensPerBlock(tokensPerBlock);
    modelConfig.setQuantMode(quantMode);
    modelConfig.setNbKvHeads(numKvHeads);

    modelConfig.setMaxBatchSize(maxBatchSize);
    modelConfig.setMaxInputLen(maxInputLen);
    modelConfig.setMaxOutputLen(maxOutputLen);

    return GptJsonConfig{name, precision, tensorParallelism, pipelineParallelism, modelConfig};
}

} // namespace

std::string GptJsonConfig::engineFilename(WorldConfig const& worldConfig, std::string const& model) const
{
    TLLM_CHECK_WITH_INFO(getTensorParallelism() == worldConfig.getTensorParallelism(), "tensor parallelism mismatch");
    TLLM_CHECK_WITH_INFO(
        getPipelineParallelism() == worldConfig.getPipelineParallelism(), "pipeline parallelism mismatch");
    auto pp = worldConfig.isPipelineParallel() ? "_pp" + std::to_string(worldConfig.getPipelineParallelism()) : "";
    return model + "_" + getPrecision() + "_tp" + std::to_string(worldConfig.getTensorParallelism()) + pp + "_rank"
        + std::to_string(worldConfig.getRank()) + ".engine";
}

GptJsonConfig GptJsonConfig::parse(std::string const& json)
{
    return parseJson(json);
}

GptJsonConfig GptJsonConfig::parse(std::istream& json)
{
    return parseJson(json);
}

GptJsonConfig GptJsonConfig::parse(std::filesystem::path const& path)
{
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(path), std::string("File does not exist: ") + path.string());
    std::ifstream json(path);
    return parse(json);
}
