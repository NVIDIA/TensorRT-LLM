#include "engines.h"

#include "tensorrt_llm/batch_manager/transformerBuffers.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInfer.h>
#include <algorithm>
#include <memory>

nvinfer1::ITensor& tensorrt_llm::testing::utils::engines::details::addInputIds(
    EngineBuildState& buildState, runtime::SizeType32 maxNumTokens)
{
    auto* input_ids = buildState.networkDefinition->addInput(batch_manager::RuntimeBuffers::kInputIdsTensorName,
        nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({-1}));
    buildState.tensors.push_back(input_ids);
    buildState.profile->setDimensions(batch_manager::RuntimeBuffers::kInputIdsTensorName,
        nvinfer1::OptProfileSelector::kMAX, runtime::ITensor::makeShape({maxNumTokens}));
    buildState.profile->setDimensions(batch_manager::RuntimeBuffers::kInputIdsTensorName,
        nvinfer1::OptProfileSelector::kOPT, runtime::ITensor::makeShape({maxNumTokens / 2}));
    buildState.profile->setDimensions(batch_manager::RuntimeBuffers::kInputIdsTensorName,
        nvinfer1::OptProfileSelector::kMIN, runtime::ITensor::makeShape({1}));
    return *input_ids;
}

nvinfer1::ITensor* tensorrt_llm::testing::utils::engines::details::addLastTokenIds(
    EngineBuildState& buildState, runtime::SizeType32 maxBatchSize, runtime::SizeType32 maxBeamWidth)
{
    auto* last_token_ids
        = buildState.networkDefinition->addInput(batch_manager::RuntimeBuffers::kLastTokenIdsTensorName,
            nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({-1}));
    buildState.tensors.push_back(last_token_ids);
    buildState.profile->setDimensions(batch_manager::RuntimeBuffers::kLastTokenIdsTensorName,
        nvinfer1::OptProfileSelector::kMAX, runtime::ITensor::makeShape({maxBatchSize * maxBeamWidth}));
    buildState.profile->setDimensions(batch_manager::RuntimeBuffers::kLastTokenIdsTensorName,
        nvinfer1::OptProfileSelector::kOPT, runtime::ITensor::makeShape({maxBatchSize * maxBeamWidth / 2}));
    buildState.profile->setDimensions(batch_manager::RuntimeBuffers::kLastTokenIdsTensorName,
        nvinfer1::OptProfileSelector::kMIN, runtime::ITensor::makeShape({1}));
    return last_token_ids;
}

nvinfer1::ITensor& tensorrt_llm::testing::utils::engines::details::addKvCacheOffsets(EngineBuildState& buildState,
    runtime::SizeType32 numPools, runtime::SizeType32 tokensPerBlock, runtime::SizeType32 maxBatchSize,
    runtime::SizeType32 maxNumTokens, runtime::SizeType32 maxBeamWidth)
{
    auto* kvCacheOffsets = buildState.networkDefinition->addInput(
        batch_manager::TransformerBuffers::kKvCacheBlockOffsetsTensorName, nvinfer1::DataType::kINT32,
        runtime::ITensor::makeShape({numPools, -1, 2, -1})); // [numPools, maxBatch * maxBeamWidth, 2, maxBlocksPerSeq]
    buildState.tensors.push_back(kvCacheOffsets);
    auto const maxBlocksPerSeq = maxNumTokens / tokensPerBlock;
    buildState.profile->setDimensions(batch_manager::TransformerBuffers::kKvCacheBlockOffsetsTensorName,
        nvinfer1::OptProfileSelector::kMAX,
        runtime::ITensor::makeShape({numPools, maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq}));
    buildState.profile->setDimensions(batch_manager::TransformerBuffers::kKvCacheBlockOffsetsTensorName,
        nvinfer1::OptProfileSelector::kOPT,
        runtime::ITensor::makeShape({numPools, maxBatchSize * maxBeamWidth / 2, 2, maxBlocksPerSeq / 2}));
    buildState.profile->setDimensions(batch_manager::TransformerBuffers::kKvCacheBlockOffsetsTensorName,
        nvinfer1::OptProfileSelector::kMIN, runtime::ITensor::makeShape({numPools, 1, 2, 1}));
    return *kvCacheOffsets;
}

tensorrt_llm::testing::utils::engines::details::EngineBuildState
tensorrt_llm::testing::utils::engines::initializeEngineBuild(std::shared_ptr<runtime::TllmLogger> const& logger)
{
    auto* builder = nvinfer1::createInferBuilder(*logger);
    auto* profile = builder->createOptimizationProfile();
    auto* network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    return {builder, network, profile, config};
}

nvinfer1::ITensor& tensorrt_llm::testing::utils::engines::details::addSingleOutputLayer(
    tensorrt_llm::testing::utils::engines::details::EngineBuildState& buildState, nvinfer1::ILayer* layer)
{
    buildState.layers.push_back(layer);
    auto* output = layer->getOutput(0);
    buildState.tensors.push_back(output);
    TLLM_LOG_INFO("Adding layer %s with output shape %s.", layer->getName(),
        tensorrt_llm::runtime::ITensor::toString(output->getDimensions()).c_str());

    return *output;
}

tensorrt_llm::common::OptionalRef<nvinfer1::ITensor> tensorrt_llm::testing::utils::engines::details::getTensorByName(
    tensorrt_llm::testing::utils::engines::details::EngineBuildState& buildState, std::string_view name)
{
    auto result = std::find_if(buildState.tensors.begin(), buildState.tensors.end(),
        [name](auto const tensor) { return tensor->getName() == name; });
    if (result == buildState.tensors.end())
    {
        return tensorrt_llm::common::OptionalRef<nvinfer1::ITensor>{};
    }
    return **result;
}
