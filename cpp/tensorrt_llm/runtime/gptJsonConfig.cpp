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

#include "common.h"
#include "gptModelConfig.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>
#include <utility>

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

GptModelConfig createModelConfig(
    Json const& json, bool engineVersionNone, SizeType tensorParallelism, nvinfer1::DataType dataType)
{
    auto const& config = engineVersionNone ? json.at("builder_config") : json.at("pretrained_config");

    auto const* const numLayersField = engineVersionNone ? "num_layers" : "num_hidden_layers";
    auto const* const numHeadsField = engineVersionNone ? "num_heads" : "num_attention_heads";
    auto const* const numKvHeadsField = engineVersionNone ? "num_kv_heads" : "num_key_value_heads";
    auto const* const mlpHiddenSizeField = engineVersionNone ? "mlp_hidden_size" : "intermediate_size";

    auto const numLayers = config.at(numLayersField).template get<SizeType>();
    auto const numHeads = config.at(numHeadsField).template get<SizeType>() / tensorParallelism;

    auto const vocabSize = config.at("vocab_size").template get<SizeType>();
    auto const hiddenSize = config.at("hidden_size").template get<SizeType>() / tensorParallelism;
    auto const sizePerHead = parseJsonFieldOr(config, "head_size", hiddenSize / numHeads);

    // TODO:
    // Code crashes when numKvHeads <= 0. Clamping downwards to 1 prevents that, make sure this is best fix.
    auto const numKvHeads
        = std::max(parseJsonFieldOr(config, numKvHeadsField, numHeads * tensorParallelism) / tensorParallelism, 1);

    auto const mlpHiddenSize = parseJsonFieldOptional<SizeType>(config, mlpHiddenSizeField);

    auto modelConfig = GptModelConfig{vocabSize, numLayers, numHeads, hiddenSize, dataType};
    modelConfig.setSizePerHead(sizePerHead);
    modelConfig.setNbKvHeads(numKvHeads);

    if (mlpHiddenSize.has_value())
    {
        modelConfig.setMlpHiddenSize(mlpHiddenSize.value() / tensorParallelism);
    }

    return modelConfig;
};

void parseBuilderConfig(GptModelConfig& modelConfig, Json const& builderConfig)
{
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

    modelConfig.setMaxBatchSize(maxBatchSize);
    modelConfig.setMaxBeamWidth(maxBeamWidth);
    modelConfig.setMaxInputLen(maxInputLen);
    modelConfig.setMaxSequenceLen(maxSequenceLen);
    modelConfig.setMaxNumTokens(maxNumTokens);
    modelConfig.setMaxDraftLen(maxDraftLen);
    modelConfig.setMaxPromptEmbeddingTableSize(maxPromptEmbeddingTableSize);
    modelConfig.computeContextLogits(computeContextLogits);
    modelConfig.computeGenerationLogits(computeGenerationLogits);
}

void parsePluginConfig(GptModelConfig& modelConfig, Json const& pluginConfig)
{
    auto const useGptAttentionPlugin = !pluginConfig.at("gpt_attention_plugin").is_null();
    auto const removeInputPadding = pluginConfig.at("remove_input_padding").template get<bool>();
    auto const& pagedKvCache = pluginConfig.at("paged_kv_cache");
    auto const& tokensPerBlock = pluginConfig.at("tokens_per_block");
    auto const useCustomAllReduce = pluginConfig.at("use_custom_all_reduce").template get<bool>();
    auto const useContextFMHAForGeneration = pluginConfig.at("use_context_fmha_for_generation").template get<bool>();
    auto const pagedContextFMHA = pluginConfig.at("use_paged_context_fmha").template get<bool>();

    modelConfig.useGptAttentionPlugin(useGptAttentionPlugin);
    modelConfig.usePackedInput(removeInputPadding);
    modelConfig.usePagedKvCache(pagedKvCache);
    modelConfig.setTokensPerBlock(tokensPerBlock);
    modelConfig.useCustomAllReduce(useCustomAllReduce);
    modelConfig.setUseContextFMHAForGeneration(useContextFMHAForGeneration);
    modelConfig.setPagedContextFMHA(pagedContextFMHA);
}

void parseLora(GptModelConfig& modelConfig, Json const& json, Json const& pluginConfig, bool engineVersionNone,
    SizeType tensorParallelism)
{
    auto const& config = engineVersionNone ? json.at("builder_config") : json.at("pretrained_config");

    auto const loraMaxRank = parseJsonFieldOr(config, "max_lora_rank", SizeType{0});
    auto const loraTargetModules = parseJsonFieldOptional<std::vector<std::string>>(config, "lora_target_modules");

    if (loraTargetModules.has_value())
    {

        modelConfig.setLoraModules(LoraModule::createLoraModules(loraTargetModules.value(), modelConfig.getHiddenSize(),
            modelConfig.getMlpHiddenSize(), modelConfig.getNbHeads(), modelConfig.getNbKvHeads(),
            modelConfig.getSizePerHead(), tensorParallelism));
    }

    modelConfig.setMaxLoraRank(loraMaxRank);

    auto useLoraPlugin = !pluginConfig.at("lora_plugin").is_null();
    if (useLoraPlugin)
    {

        if (modelConfig.getLoraModules().empty() || modelConfig.getMaxLoraRank() == 0)
        {
            TLLM_LOG_WARNING("lora_plugin enabled, but no lora module enabled: setting useLoraPlugin to false");
            useLoraPlugin = false;
        }
    }
    modelConfig.useLoraPlugin(useLoraPlugin);
}

template <typename InputType>
GptJsonConfig parseJson(InputType&& input)
{
    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto const json = nlohmann::json::parse(std::forward<InputType>(input), nullptr, allowExceptions, ingoreComments);

    auto const engineVersion = parseJsonFieldOr(json, "version", std::string("none"));

    auto const engineVersionNone = engineVersion == std::string("none");
    if (engineVersionNone)
    {
        TLLM_LOG_INFO("No engine version found in the config file, assuming engine(s) built by old builder API.");
    }
    else
    {
        TLLM_LOG_INFO("Engine version %s found in the config file, assuming engine(s) built by new builder API.",
            engineVersion.c_str());
    }

    auto const& builderConfig = engineVersionNone ? json.at("builder_config") : json.at("build_config");

    auto const name = engineVersionNone ? builderConfig.at("name").template get<std::string>()
                                        : json.at("pretrained_config").at("architecture").template get<std::string>();

    auto const tensorParallelism = engineVersionNone
        ? builderConfig.at("tensor_parallel").template get<SizeType>()
        : json.at("pretrained_config").at("mapping").at("tp_size").template get<SizeType>();
    auto const pipelineParallelism = engineVersionNone
        ? parseJsonFieldOr(builderConfig, "pipeline_parallel", 1)
        : parseJsonFieldOr(json.at("pretrained_config").at("mapping"), "pp_size", 1);

    auto const precision = engineVersionNone ? builderConfig.at("precision").template get<std::string>()
                                             : json.at("pretrained_config").at("dtype").template get<std::string>();

    auto const dataType = [&precision]()
    {
        if (!precision.compare("float32"))
            return nvinfer1::DataType::kFLOAT;
        else if (!precision.compare("float16"))
            return nvinfer1::DataType::kHALF;
        else if (!precision.compare("bfloat16"))
            return nvinfer1::DataType::kBF16;
        else
            TLLM_THROW("Model data type '%s' not supported", precision.c_str());
    }();

    auto modelConfig = createModelConfig(json, engineVersionNone, tensorParallelism, dataType);

    parseBuilderConfig(modelConfig, builderConfig);

    auto const& pluginConfig = engineVersionNone ? json.at("plugin_config") : builderConfig.at("plugin_config");
    parsePluginConfig(modelConfig, pluginConfig);

    parseLora(modelConfig, json, pluginConfig, engineVersionNone, tensorParallelism);

    if (engineVersionNone)
    {
        auto const quantMode
            = tc::QuantMode(parseJsonFieldOr(builderConfig, "quant_mode", tc::QuantMode::none().value()));
        modelConfig.setQuantMode(quantMode);
    }
    else
    {
        auto const& quantization = json.at("pretrained_config").at("quantization");
        auto quantAlgo = parseJsonFieldOptional<std::string>(quantization, "quant_algo");
        auto kvCacheQuantAlgo = parseJsonFieldOptional<std::string>(quantization, "kv_cache_quant_algo");
        auto const quantMode = tc::QuantMode::fromQuantAlgo(quantAlgo, kvCacheQuantAlgo);
        modelConfig.setQuantMode(quantMode);
    }

    if (engineVersionNone)
    {
        if (name == std::string("chatglm_6b") || name == std::string("glm_10b"))
        {
            modelConfig.setModelVariant(GptModelConfig::ModelVariant::kGlm);
            // kGlm is only for ChatGLM-6B and GLM-10B
        }
    }
    else
    {
        if (name == "ChatGLMForCausalLM")
        {
            auto const& pretrainedConfig = json.at("pretrained_config");
            auto const chatglmVersion = pretrainedConfig.at("chatglm_version").template get<std::string>();
            if (chatglmVersion == "glm" || chatglmVersion == "chatglm")
            {
                modelConfig.setModelVariant(GptModelConfig::ModelVariant::kGlm);
                // kGlm is only for ChatGLM-6B and GLM-10B
            }
        }
    }

    if (!engineVersionNone)
    {
        auto const& pretrainedConfig = json.at("pretrained_config");
        auto const medusaHeads = parseJsonFieldOptional<SizeType>(pretrainedConfig, "num_medusa_heads");
        auto const maxDraftLen = parseJsonFieldOptional<SizeType>(pretrainedConfig, "max_draft_len");
        TLLM_CHECK_WITH_INFO((medusaHeads.has_value() ^ maxDraftLen.has_value()) == 0,
            "Either both num_medusa_heads and max_draft_len or none have to be provided");
        if (medusaHeads.has_value() && medusaHeads.value() > 0)
        {
            modelConfig.setMaxDraftLen(maxDraftLen.value());
            auto medusaModule = MedusaModule(medusaHeads.value(), maxDraftLen.value());
            modelConfig.setMedusaModule(medusaModule);
        }
    }

    return GptJsonConfig{name, engineVersion, precision, tensorParallelism, pipelineParallelism, modelConfig};
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
