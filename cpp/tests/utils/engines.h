#ifndef CA1B91B5_DF64_4CF8_948F_5AFF243A2555
#define CA1B91B5_DF64_4CF8_948F_5AFF243A2555

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <tensorrt_llm/batch_manager/runtimeBuffers.h>
#include <utility>
#include <vector>

namespace tensorrt_llm::testing::utils::engines
{

namespace details
{

struct EngineBuildResource
{
    EngineBuildResource() = default;
    virtual ~EngineBuildResource() = default;
    EngineBuildResource(EngineBuildResource const& vector) = default;
    EngineBuildResource& operator=(EngineBuildResource const& vector) = default;
    EngineBuildResource(EngineBuildResource&& vector) noexcept = default;
    EngineBuildResource& operator=(EngineBuildResource&& vector) noexcept = default;
};

template <typename TValue>
struct Vector : public EngineBuildResource
{
    explicit Vector(std::vector<TValue> values)
        : values(std::move(values)){};
    Vector(Vector const& vector) = default;
    Vector& operator=(Vector const& vector) = default;
    Vector(Vector&& vector) noexcept = default;
    Vector& operator=(Vector&& vector) noexcept = default;
    ~Vector() override = default;
    std::vector<TValue> values;
};

template <typename TValue, size_t Size>
struct Array : public EngineBuildResource
{
    explicit Array(std::array<TValue, Size> values)
        : values(std::move(values)){};
    Array(Array const& vector) = default;
    Array& operator=(Array const& vector) = default;
    Array(Array&& vector) noexcept = default;
    Array& operator=(Array&& vector) noexcept = default;
    ~Array() override = default;
    std::array<TValue, Size> values;
};

struct EngineBuildState
{
    EngineBuildState(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* networkDefinition,
        nvinfer1::IOptimizationProfile* profile, nvinfer1::IBuilderConfig* builderConfig)
        : builder(builder)
        , networkDefinition(networkDefinition)
        , profile(profile)
        , builderConfig(builderConfig){};
    EngineBuildState(EngineBuildState const& vector) = delete;
    EngineBuildState& operator=(EngineBuildState const& vector) = delete;
    EngineBuildState(EngineBuildState&& vector) noexcept = default;
    EngineBuildState& operator=(EngineBuildState&& vector) noexcept = default;
    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> networkDefinition;
    nvinfer1::IOptimizationProfile* profile;

    // While building the engine, one might need some data for weights and such. Turns out, TensorRT does not keep a
    // copy of those, so if you create them as temporaries and pass them to the TRT APIs, you will get UB. So we need
    // some place where we can keep those things.
    std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig;
    std::vector<std::unique_ptr<EngineBuildResource>> resources;
    std::vector<nvinfer1::ITensor*> tensors;
    std::vector<nvinfer1::ILayer*> layers;

    ~EngineBuildState()
    {
        // Builder needs to be deleteds last.
        networkDefinition.reset();
        builderConfig.reset();
        builder.reset();
    }
};

common::OptionalRef<nvinfer1::ITensor> getTensorByName(EngineBuildState& buildState, std::string_view name);

nvinfer1::ITensor& addSingleOutputLayer(EngineBuildState& buildState, nvinfer1::ILayer* layer);

template <typename TResource>
TResource& addResource(EngineBuildState& buildState, TResource resource)
{
    return *dynamic_cast<TResource*>(
        buildState.resources.emplace_back(std::make_unique<TResource>(std::move(resource))).get());
}

template <typename TValue>
Vector<TValue>& addSingleConstantVectorResource(EngineBuildState& buildState, TValue value, std::size_t length)
{
    std::vector<TValue> weights(length);
    std::fill(weights.begin(), weights.end(), value);
    return addResource(buildState, Vector<TValue>{weights});
}

template <typename TValue>
Vector<TValue>& addConstantVectorResource(EngineBuildState& buildState, std::vector<TValue> values)
{
    return addResource(buildState, Vector<TValue>{values});
}

template <typename TValue>
Array<TValue, 1>& addConstantScalarResource(EngineBuildState& buildState, TValue value)
{
    return addResource(buildState, Array<TValue, 1>{{value}});
}

nvinfer1::ITensor& addInputIds(EngineBuildState& buildState, runtime::SizeType32 maxNumTokens);
nvinfer1::ITensor* addLastTokenIds(
    EngineBuildState& buildState, runtime::SizeType32 maxBatchSize, runtime::SizeType32 maxBeamWidth);
nvinfer1::ITensor& addKvCacheOffsets(EngineBuildState& buildState, runtime::SizeType32 numPools,
    runtime::SizeType32 tokensPerBlock, runtime::SizeType32 maxBatchSize, runtime::SizeType32 maxNumTokens,
    runtime::SizeType32 maxBeamWidth);

template <typename TValue>
nvinfer1::ITensor& addSingleConstantVector(EngineBuildState& buildState, TValue value, runtime::SizeType32 length)
{
    auto& resourceWeights = addSingleConstantVectorResource(buildState, value, length);
    auto const trtDatatype = runtime::TRTDataType<TValue>::value;
    auto* layer = buildState.networkDefinition->addConstant(runtime::ITensor::makeShape({length}),
        {trtDatatype, resourceWeights.values.data(), static_cast<runtime::ITensor::DimType64>(length)});
    return addSingleOutputLayer(buildState, layer);
}

template <typename TValue>
nvinfer1::ITensor& addSingleConstantTensor(EngineBuildState& buildState, TValue value, runtime::SizeType32 length)
{
    auto& resourceWeights = addSingleConstantVectorResource(buildState, value, length);
    auto const trtDatatype = runtime::TRTDataType<TValue>::value;
    auto* layer = buildState.networkDefinition->addConstant(runtime::ITensor::makeShape({1, length}),
        {trtDatatype, resourceWeights.values.data(), static_cast<runtime::ITensor::DimType64>(length)});
    return addSingleOutputLayer(buildState, layer);
}

template <typename TValue>
nvinfer1::ITensor& addConstantVector(EngineBuildState& buildState, std::vector<TValue> values)
{
    auto& resourceWeights = addConstantVectorResource(buildState, values);
    auto const trtDatatype = runtime::TRTDataType<TValue>::value;
    auto const length = static_cast<runtime::ITensor::DimType64>(values.size());
    auto* layer = buildState.networkDefinition->addConstant(
        runtime::ITensor::makeShape({length}), {trtDatatype, resourceWeights.values.data(), length});
    return addSingleOutputLayer(buildState, layer);
}

template <typename TValue>
nvinfer1::ITensor& addConstantTensor(
    EngineBuildState& buildState, std::vector<TValue> values, runtime::ITensor::Shape shape)
{
    auto& resourceWeights = addConstantVectorResource(buildState, values);
    auto const trtDatatype = runtime::TRTDataType<TValue>::value;
    auto const count = runtime::ITensor::volume(shape);
    auto* layer = buildState.networkDefinition->addConstant(shape, {trtDatatype, resourceWeights.values.data(), count});
    return addSingleOutputLayer(buildState, layer);
}

template <typename TValue>
nvinfer1::ITensor& addSingleConstantTensor(EngineBuildState& buildState, TValue value, runtime::ITensor::Shape shape)
{
    auto const count = runtime::ITensor::volume(shape);
    auto& resourceWeights = addSingleConstantVectorResource(buildState, value, count);
    auto const trtDatatype = runtime::TRTDataType<TValue>::value;
    auto* layer = buildState.networkDefinition->addConstant(shape, {trtDatatype, resourceWeights.values.data(), count});
    return addSingleOutputLayer(buildState, layer);
}

template <typename TValue>
nvinfer1::ITensor& addConstantScalar(EngineBuildState& buildState, TValue value)
{
    auto& resourceWeights = addConstantScalarResource<TValue>(buildState, value);
    auto const trtDatatype = runtime::TRTDataType<TValue>::value;
    auto* layer = buildState.networkDefinition->addConstant(
        runtime::ITensor::makeShape({}), {trtDatatype, resourceWeights.values.data(), 1});
    return addSingleOutputLayer(buildState, layer);
}

template <typename TValue>
nvinfer1::ITensor& oneHotEncode(
    EngineBuildState& buildState, nvinfer1::ITensor& inputIds, runtime::SizeType32 vocabSize)
{
    auto const trtValueType = runtime::TRTDataType<TValue>::value;
    auto& oneHotValues = addConstantVector<TValue>(buildState, {0, 1});
    auto& oneHotDepth = addConstantScalar(buildState, vocabSize);
    auto* oneHotLayer = buildState.networkDefinition->addOneHot(inputIds, oneHotValues, oneHotDepth, 0);
    return addSingleOutputLayer(buildState, oneHotLayer);
}
} // namespace details

struct TrivialDecoderParameters
{
    TrivialDecoderParameters(runtime::SizeType32 vocabSize, runtime::SizeType32 maxBatchSize,
        runtime::SizeType32 maxNumTokens, runtime::SizeType32 tokensPerBlock, runtime::SizeType32 maxBeamWidth,
        bool gatherContextLogits)
        : vocabSize(vocabSize)
        , maxBatchSize(maxBatchSize)
        , maxNumTokens(maxNumTokens)
        , tokensPerBlock(tokensPerBlock)
        , maxBeamWidth(maxBeamWidth)
        , gatherContextLogits(gatherContextLogits){};
    runtime::SizeType32 vocabSize;
    runtime::SizeType32 maxBatchSize;
    runtime::SizeType32 maxNumTokens;
    runtime::SizeType32 tokensPerBlock;
    runtime::SizeType32 maxBeamWidth;
    bool gatherContextLogits;
};

details::EngineBuildState initializeEngineBuild(std::shared_ptr<runtime::TllmLogger> const& logger);

template <typename TLogits>
std::unique_ptr<nvinfer1::IHostMemory> createTrivialDecoder(
    TrivialDecoderParameters parameters, std::shared_ptr<runtime::TllmLogger> const& logger)
{
    auto const trtLogitsType = runtime::TRTDataType<TLogits>::value;
    auto buildState = initializeEngineBuild(logger);
    auto* builder = buildState.builder.get();
    auto* profile = buildState.profile;
    auto* network = buildState.networkDefinition.get();
    auto& inputIds = details::addInputIds(buildState, parameters.maxNumTokens);
    auto& kvCacheOffsets = details::addKvCacheOffsets(buildState, 1, parameters.tokensPerBlock, parameters.maxBatchSize,
        parameters.maxNumTokens, parameters.maxBeamWidth);

    auto& oneHotLayerOutput = details::oneHotEncode<TLogits>(buildState, inputIds, parameters.vocabSize);
    oneHotLayerOutput.setName(batch_manager::RuntimeBuffers::kLogitsTensorName);
    network->markOutput(oneHotLayerOutput);

    buildState.builderConfig->addOptimizationProfile(profile);
    buildState.builderConfig->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    auto* engine = builder->buildSerializedNetwork(*network, *buildState.builderConfig);
    return std::unique_ptr<nvinfer1::IHostMemory>(engine);
}

template <typename TLogits>
struct ConstantTrivialDecoderParameters
{
    ConstantTrivialDecoderParameters(TrivialDecoderParameters trivialDecoderParameters, std::vector<TLogits> logits)
        : trivialDecoderParameters(trivialDecoderParameters)
        , logits(logits)
    {
        auto const sizeTypeVocabSize = static_cast<std::size_t>(trivialDecoderParameters.vocabSize);
        auto const logitsSize = logits.size();
        TLLM_CHECK_WITH_INFO(static_cast<std::size_t>(trivialDecoderParameters.vocabSize) == logits.size(),
            "The size of the constant logits (%lu) has to be equal to the vocabulary size (%lu).", logitsSize,
            sizeTypeVocabSize);
    };

    TrivialDecoderParameters trivialDecoderParameters;
    std::vector<TLogits> logits;
};

template <typename TLogits>
details::EngineBuildState createConstantTrivialDecoderBase(
    ConstantTrivialDecoderParameters<TLogits> parameters, std::shared_ptr<runtime::TllmLogger> const& logger)
{
    auto const trtLogitsType = runtime::TRTDataType<TLogits>::value;
    auto buildState = initializeEngineBuild(logger);
    auto* builder = buildState.builder.get();
    auto* profile = buildState.profile;
    auto* network = buildState.networkDefinition.get();
    auto& inputIds = details::addInputIds(buildState, parameters.trivialDecoderParameters.maxNumTokens);
    nvinfer1::ITensor* lastTokenIds = nullptr;
    if (!parameters.trivialDecoderParameters.gatherContextLogits)
    {
        lastTokenIds = details::addLastTokenIds(buildState, parameters.trivialDecoderParameters.maxBatchSize,
            parameters.trivialDecoderParameters.maxBeamWidth);
    }
    auto& kvCacheOffsets = details::addKvCacheOffsets(buildState, 1, parameters.trivialDecoderParameters.tokensPerBlock,
        parameters.trivialDecoderParameters.maxBatchSize, parameters.trivialDecoderParameters.maxNumTokens,
        parameters.trivialDecoderParameters.maxBeamWidth);

    auto const vocabSize = static_cast<runtime::ITensor::DimType64>(parameters.logits.size());

    auto& constantLogitsPerToken = details::addConstantTensor<TLogits>(
        buildState, parameters.logits, runtime::ITensor::makeShape({vocabSize, 1}));
    auto& oneHotLayerOutput
        = details::oneHotEncode<TLogits>(buildState, inputIds, parameters.trivialDecoderParameters.vocabSize);
    auto& ones = details::addSingleConstantTensor<TLogits>(buildState, 1, runtime::ITensor::makeShape({1, vocabSize}));
    auto* intermediateLayer1 = network->addMatrixMultiply(
        ones, nvinfer1::MatrixOperation::kNONE, oneHotLayerOutput, nvinfer1::MatrixOperation::kNONE);
    auto* intermediateLayer1Output = intermediateLayer1->getOutput(0);

    nvinfer1::ITensor* gatherLayerOutput = nullptr;
    if (!parameters.trivialDecoderParameters.gatherContextLogits)
    {
        auto& one = details::addSingleConstantTensor<int32_t>(buildState, 1, runtime::ITensor::makeShape({1}));
        auto* lastTokenIdsMinus1Layer
            = network->addElementWise(*lastTokenIds, one, nvinfer1::ElementWiseOperation::kSUB);
        auto* gatherLayer = network->addGather(*intermediateLayer1Output, *lastTokenIdsMinus1Layer->getOutput(0), 1);
        gatherLayerOutput = gatherLayer->getOutput(0);
    }
    else
    {
        gatherLayerOutput = intermediateLayer1Output;
    }

    auto* constLogitsLayer = network->addMatrixMultiply(*gatherLayerOutput, nvinfer1::MatrixOperation::kTRANSPOSE,
        constantLogitsPerToken, nvinfer1::MatrixOperation::kTRANSPOSE);
    auto* outputLogits = constLogitsLayer->getOutput(0);
    network->markOutput(*outputLogits);
    outputLogits->setName(batch_manager::RuntimeBuffers::kLogitsTensorName);
    buildState.tensors.push_back(outputLogits);
    return buildState;
}

template <typename TLogits>
std::unique_ptr<nvinfer1::IHostMemory> createConstantTrivialDecoder(
    ConstantTrivialDecoderParameters<TLogits> parameters, std::shared_ptr<runtime::TllmLogger> const& logger)
{
    auto buildState = createConstantTrivialDecoderBase<TLogits>(parameters, logger);
    buildState.builderConfig->addOptimizationProfile(buildState.profile);
    buildState.builderConfig->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    auto* engine = buildState.builder->buildSerializedNetwork(*buildState.networkDefinition, *buildState.builderConfig);
    return std::unique_ptr<nvinfer1::IHostMemory>(engine);
}

template <typename TLogits>
std::unique_ptr<nvinfer1::IHostMemory> createConstantTrivialDecoderWithTopKLogits(
    ConstantTrivialDecoderParameters<TLogits> parameters, runtime::SizeType32 numTopLogits, std::string_view outputName,
    std::shared_ptr<runtime::TllmLogger> const& logger)
{
    auto buildState = createConstantTrivialDecoderBase<TLogits>(parameters, logger);
    auto logits = details::getTensorByName(buildState, batch_manager::RuntimeBuffers::kLogitsTensorName);
    TLLM_CHECK_WITH_INFO(static_cast<bool>(logits),
        "You can only add topk logits on top of a network which contains a tensor named %s",
        batch_manager::RuntimeBuffers::kLogitsTensorName);
    auto* topKLayer = buildState.networkDefinition->addTopK(
        logits.value(), nvinfer1::TopKOperation::kMAX, numTopLogits, 1UL << 1UL);
    auto* topKLayerOutput = topKLayer->getOutput(0);
    topKLayerOutput->setName(outputName.data());
    buildState.networkDefinition->markOutput(*topKLayerOutput);
    auto* profile = buildState.profile;
    buildState.builderConfig->addOptimizationProfile(profile);
    buildState.builderConfig->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    auto* engine = buildState.builder->buildSerializedNetwork(*buildState.networkDefinition, *buildState.builderConfig);
    return std::unique_ptr<nvinfer1::IHostMemory>(engine);
}
} // namespace tensorrt_llm::testing::utils::engines

#endif /* CA1B91B5_DF64_4CF8_948F_5AFF243A2555 */
