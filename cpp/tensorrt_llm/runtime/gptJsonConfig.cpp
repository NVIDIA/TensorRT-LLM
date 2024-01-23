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

#include "tensorrt_llm/runtime/gptJsonConfig.h"

#include "loraManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
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
    catch (nlohmann::json::out_of_range& e)
    {
        TLLM_LOG_WARNING("Parameter %s cannot be read from json:", std::string(name).c_str());
        TLLM_LOG_WARNING(e.what());
    }
    return value;
}

template <typename FieldType>
std::optional<FieldType> parseJsonFieldOptional(Json const& json, std::string_view name)
{
    std::optional<FieldType> value = std::nullopt;
    try
    {
        value = json.at(name).template get<FieldType>();
    }
    catch (const nlohmann::json::out_of_range& e)
    {
        TLLM_LOG_WARNING(e.what());
        TLLM_LOG_WARNING("Optional value for parameter %s will not be set.", std::string(name).c_str());
    }
    catch (const nlohmann::json::type_error& e)
    {
        TLLM_LOG_WARNING(e.what());
        TLLM_LOG_WARNING("Optional value for parameter %s will not be set.", std::string(name).c_str());
    }
    return value;
}

template <typename InputType>
GptJsonConfig parseJson(InputType&& i)
{
    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto json = nlohmann::json::parse(i, nullptr, allowExceptions, ingoreComments);

    auto engine_version = parseJsonFieldOr(json, "version", std::string("none"));
    if (engine_version == std::string("none"))
    {
        auto const& builderConfig = json.at("builder_config");
        auto const name = builderConfig.at("name").template get<std::string>();
        auto const precision = builderConfig.at("precision").template get<std::string>();
        auto const tensorParallelism = builderConfig.at("tensor_parallel").template get<SizeType>();
        auto const pipelineParallelism = parseJsonFieldOr(builderConfig, "pipeline_parallel", 1);
        auto const numHeads = builderConfig.at("num_heads").template get<SizeType>() / tensorParallelism;
        auto const hiddenSize = builderConfig.at("hidden_size").template get<SizeType>() / tensorParallelism;
        auto const mlpHiddenSize = parseJsonFieldOr(builderConfig, "mlp_hidden_size", SizeType{0}) / tensorParallelism;
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

        auto const quantMode
            = tc::QuantMode(parseJsonFieldOr(builderConfig, "quant_mode", tc::QuantMode::none().value()));
        // TODO:
        // Code crashes when numKvHeads <= 0. Clamping downwards to 1 prevents that, make sure this is best fix.
        auto const numKvHeads = std::max(
            parseJsonFieldOr(builderConfig, "num_kv_heads", numHeads * tensorParallelism) / tensorParallelism, 1);
        auto const maxBatchSize = parseJsonFieldOr(builderConfig, "max_batch_size", 0);
        auto const maxBeamWidth = parseJsonFieldOr(builderConfig, "max_beam_width", 0);
        auto const maxInputLen = parseJsonFieldOr(builderConfig, "max_input_len", 0);
        auto const maxSequenceLen = maxInputLen + parseJsonFieldOr(builderConfig, "max_output_len", 0);
        auto const maxDraftLen = parseJsonFieldOr(builderConfig, "max_draft_len", 0);
        auto const maxNumTokens = parseJsonFieldOptional<SizeType>(builderConfig, "max_num_tokens");
        auto const maxPromptEmbeddingTableSize
            = parseJsonFieldOr<SizeType>(builderConfig, "max_prompt_embedding_table_size", 0);

        auto const computeContextLogits = parseJsonFieldOr(builderConfig, "gather_context_logits", false);
        auto const computeGenerationLogits = parseJsonFieldOr(builderConfig, "gather_generation_logits", false);
        ;

        auto const& pluginConfig = json.at("plugin_config");
        auto const pagedKvCache = pluginConfig.at("paged_kv_cache");
        auto const tokensPerBlock = pluginConfig.at("tokens_per_block");
        auto const useGptAttentionPlugin = !pluginConfig.at("gpt_attention_plugin").is_null();
        auto const removeInputPadding = pluginConfig.at("remove_input_padding").template get<bool>();
        auto const useCustomAllReduce = pluginConfig.at("use_custom_all_reduce").template get<bool>();
        auto const useContextFMHAForGeneration
            = pluginConfig.at("use_context_fmha_for_generation").template get<bool>();
        auto const pagedContextFMHA = pluginConfig.at("use_paged_context_fmha").template get<bool>();
        auto useLoraPlugin = !pluginConfig.at("lora_plugin").is_null();

        auto modelConfig = GptModelConfig{vocabSize, numLayers, numHeads, hiddenSize, dataType};
        modelConfig.useGptAttentionPlugin(useGptAttentionPlugin);
        modelConfig.usePackedInput(removeInputPadding);
        modelConfig.usePagedKvCache(pagedKvCache);
        modelConfig.useCustomAllReduce(useCustomAllReduce);
        modelConfig.setTokensPerBlock(tokensPerBlock);
        modelConfig.setQuantMode(quantMode);
        modelConfig.setNbKvHeads(numKvHeads);
        modelConfig.computeContextLogits(computeContextLogits);
        modelConfig.computeGenerationLogits(computeGenerationLogits);
        modelConfig.setUseContextFMHAForGeneration(useContextFMHAForGeneration);
        modelConfig.setPagedContextFMHA(pagedContextFMHA);
        modelConfig.setMlpHiddenSize(mlpHiddenSize);

        modelConfig.setMaxBatchSize(maxBatchSize);
        modelConfig.setMaxBeamWidth(maxBeamWidth);
        modelConfig.setMaxInputLen(maxInputLen);
        modelConfig.setMaxSequenceLen(maxSequenceLen);
        modelConfig.setMaxNumTokens(maxNumTokens);
        modelConfig.setMaxDraftLen(maxDraftLen);
        modelConfig.setMaxPromptEmbeddingTableSize(maxPromptEmbeddingTableSize);

        if (useLoraPlugin)
        {
            auto const loraTargetModules
                = parseJsonFieldOr(builderConfig, "lora_target_modules", std::vector<std::string>{});

            modelConfig.setLoraModules(LoraModule::createLoraModules(loraTargetModules, hiddenSize, mlpHiddenSize,
                numHeads, numKvHeads, modelConfig.getSizePerHead(), tensorParallelism));

            if (modelConfig.getLoraModules().size() == 0)
            {
                TLLM_LOG_WARNING("lora_plugin enabled, but no lora module enabled: setting useLoraPlugin to false");
                useLoraPlugin = false;
            }
        }
        modelConfig.useLoraPlugin(useLoraPlugin);

        if (name == std::string("chatglm_6b") || name == std::string("glm_10b"))
        {
            modelConfig.setModelVariant(GptModelConfig::ModelVariant::kGlm);
            // kGlm is only for ChatGLM-6B and GLM-10B
        }

        return GptJsonConfig{name, engine_version, precision, tensorParallelism, pipelineParallelism, modelConfig};
    }
    else
    {
        auto const& pretrainedConfig = json.at("pretrained_config");
        auto const& buildConfig = json.at("build_config");
        auto const architecture = pretrainedConfig.at("architecture").template get<std::string>();
        auto const name = architecture;

        auto const dtype = pretrainedConfig.at("dtype").template get<std::string>();
        auto const& mapping = pretrainedConfig.at("mapping");
        auto const tpSize = mapping.at("tp_size").template get<SizeType>();
        auto const ppSize = parseJsonFieldOr(mapping, "pp_size", 1);
        auto const numAttentionHeads = pretrainedConfig.at("num_attention_heads").template get<SizeType>() / tpSize;
        auto const hiddenSize = pretrainedConfig.at("hidden_size").template get<SizeType>() / tpSize;
        auto const vocabSize = pretrainedConfig.at("vocab_size").template get<SizeType>();
        auto const numHiddenLayers = pretrainedConfig.at("num_hidden_layers").template get<SizeType>();

        auto dataType = nvinfer1::DataType::kFLOAT;
        if (!dtype.compare("float32"))
            dataType = nvinfer1::DataType::kFLOAT;
        else if (!dtype.compare("float16"))
            dataType = nvinfer1::DataType::kHALF;
        else if (!dtype.compare("bfloat16"))
            dataType = nvinfer1::DataType::kBF16;
        else
            TLLM_CHECK_WITH_INFO(false, tc::fmtstr("Model data type '%s' not supported", dtype.c_str()));

        auto const& quantization = pretrainedConfig.at("quantization");
        auto useSmoothQuant = parseJsonFieldOr(quantization, "use_smooth_quant", false);
        auto perChannel = parseJsonFieldOr(quantization, "per_channel", false);
        auto perToken = parseJsonFieldOr(quantization, "per_token", false);
        // TODO: Unused parameters
        // auto perGroup = parseJsonFieldOr(quantization, "per_group", false);
        // auto groupSize = parseJsonFieldOr(quantization, "group_size", 128);
        auto int8KvCache = parseJsonFieldOr(quantization, "int8_kv_cache", false);
        auto enableFp8 = parseJsonFieldOr(quantization, "enable_fp8", false);
        auto fp8KvCache = parseJsonFieldOr(quantization, "fp8_kv_cache", false);
        auto useWeightOnly = parseJsonFieldOr(quantization, "use_weight_only", false);
        auto weightOnlyPrecision = parseJsonFieldOr(quantization, "weight_only_precision", std::string("int8"));
        bool quantizeWeights = false;
        bool quantizeActivations = false;
        if (useSmoothQuant)
        {
            quantizeWeights = true;
            quantizeActivations = true;
        }
        else if (useWeightOnly)
        {
            quantizeWeights = true;
            perToken = false;
            perChannel = false;
        }
        bool useInt4Weights = (weightOnlyPrecision == std::string("int4"));
        auto const quantMode = tc::QuantMode::fromDescription(quantizeWeights, quantizeActivations, perToken,
            perChannel, useInt4Weights, int8KvCache, fp8KvCache, enableFp8);

        // TODO:
        // Code crashes when numKvHeads <= 0. Clamping downwards to 1 prevents that, make sure this is best fix.
        auto const numKVHeads = pretrainedConfig.at("num_key_value_heads").template get<SizeType>();
        auto const numKeyValueHeads = std::max(numKVHeads / tpSize, 1);

        auto const maxBatchSize = parseJsonFieldOr(buildConfig, "max_batch_size", 0);
        auto const maxBeamWidth = parseJsonFieldOr(buildConfig, "max_beam_width", 0);
        auto const maxInputLen = parseJsonFieldOr(buildConfig, "max_input_len", 0);
        auto const maxSequenceLen = maxInputLen + parseJsonFieldOr(buildConfig, "max_output_len", 0);
        auto const maxDraftLen = parseJsonFieldOr(buildConfig, "max_draft_len", 0);
        auto const maxNumTokens = parseJsonFieldOptional<SizeType>(buildConfig, "max_num_tokens");
        auto const maxPromptEmbeddingTableSize
            = parseJsonFieldOr<SizeType>(buildConfig, "max_prompt_embedding_table_size", 0);
        auto const computeContextLogits = parseJsonFieldOr(buildConfig, "gather_context_logits", false);
        auto const computeGenerationLogits = parseJsonFieldOr(buildConfig, "gather_generation_logits", false);

        auto const& pluginConfig = buildConfig.at("plugin_config");
        auto const pagedKvCache = pluginConfig.at("paged_kv_cache");
        auto const tokensPerBlock = pluginConfig.at("tokens_per_block");
        auto const useGptAttentionPlugin = !pluginConfig.at("gpt_attention_plugin").is_null();
        auto const removeInputPadding = pluginConfig.at("remove_input_padding").template get<bool>();
        auto const useCustomAllReduce = pluginConfig.at("use_custom_all_reduce").template get<bool>();
        auto const useContextFMHAForGeneration
            = pluginConfig.at("use_context_fmha_for_generation").template get<bool>();

        auto modelConfig = GptModelConfig{vocabSize, numHiddenLayers, numAttentionHeads, hiddenSize, dataType};
        modelConfig.useGptAttentionPlugin(useGptAttentionPlugin);
        modelConfig.usePackedInput(removeInputPadding);
        modelConfig.usePagedKvCache(pagedKvCache);
        modelConfig.useCustomAllReduce(useCustomAllReduce);
        modelConfig.setTokensPerBlock(tokensPerBlock);
        modelConfig.setQuantMode(quantMode);
        modelConfig.setNbKvHeads(numKeyValueHeads);
        modelConfig.computeContextLogits(computeContextLogits);
        modelConfig.computeGenerationLogits(computeGenerationLogits);
        modelConfig.setUseContextFMHAForGeneration(useContextFMHAForGeneration);

        modelConfig.setMaxBatchSize(maxBatchSize);
        modelConfig.setMaxBeamWidth(maxBeamWidth);
        modelConfig.setMaxInputLen(maxInputLen);
        modelConfig.setMaxSequenceLen(maxSequenceLen);
        modelConfig.setMaxNumTokens(maxNumTokens);
        modelConfig.setMaxDraftLen(maxDraftLen);
        modelConfig.setMaxPromptEmbeddingTableSize(maxPromptEmbeddingTableSize);

        // TODO: Verify the architecture field in ChatGLM models
        if (name == std::string("ChatGLMModel") || name == std::string("GLMModel"))
        {
            modelConfig.setModelVariant(GptModelConfig::ModelVariant::kGlm);
            // kGlm is only for ChatGLM-6B and GLM-10B
        }

        return GptJsonConfig{name, engine_version, dtype, tpSize, ppSize, modelConfig};
    }
}

} // namespace

std::string GptJsonConfig::engineFilename(WorldConfig const& worldConfig, std::string const& model) const
{
    TLLM_CHECK_WITH_INFO(getTensorParallelism() == worldConfig.getTensorParallelism(), "tensor parallelism mismatch");
    TLLM_CHECK_WITH_INFO(
        getPipelineParallelism() == worldConfig.getPipelineParallelism(), "pipeline parallelism mismatch");
    auto pp = worldConfig.isPipelineParallel() ? "_pp" + std::to_string(worldConfig.getPipelineParallelism()) : "";
    if (getVersion() == std::string("none"))
    {
        return model + "_" + getPrecision() + "_tp" + std::to_string(worldConfig.getTensorParallelism()) + pp + "_rank"
            + std::to_string(worldConfig.getRank()) + ".engine";
    }
    else
    {
        return "rank" + std::to_string(worldConfig.getRank()) + ".engine";
    }
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
