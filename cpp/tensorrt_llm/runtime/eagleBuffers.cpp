/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/eagleBuffers.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"

namespace tksd = tensorrt_llm::kernels::speculative_decoding;

namespace tensorrt_llm::runtime
{

void EagleBuffers::Inputs::create(SizeType32 maxNumSequences, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto const& manager = runtime.getBufferManager();

    auto const& speculativeDecodingModule = modelConfig.getSpeculativeDecodingModule();
    auto const maxNumPaths = speculativeDecodingModule.getMaxNumPaths();
    auto const maxDraftPathLen = speculativeDecodingModule.getMaxDraftPathLen();
    auto const maxPathLen = speculativeDecodingModule.getMaxPathLen();
    auto const maxDecodingTokens = speculativeDecodingModule.getMaxDecodingTokens();
    auto const maxDecodingDraftTokens = speculativeDecodingModule.getMaxDecodingDraftTokens();

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;

    temperatures = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kFLOAT);
    randomDataSample = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kFLOAT);
    randomDataValidation
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxDraftPathLen}), nvinfer1::DataType::kFLOAT);
    draftTokens = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingDraftTokens}), TRTTokenIdType);
    draftLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    draftPaths
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxPathLen}), nvinfer1::DataType::kINT32);
    specDecodingGenerationLengths = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    specDecodingPackedMasks
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingTokens, common::ceilDiv(maxDecodingTokens, 32)}),
            nvinfer1::DataType::kINT32);
    specDecodingPositionOffsets
        = manager.gpu(ITensor::makeShape({maxNumSequences * maxDecodingTokens}), nvinfer1::DataType::kINT32);

    eagleNetCtxRequestTypesHost = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetCtxContextLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetCtxPastKeyValueLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetGenRequestTypesHost = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetGenContextLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetGenPastKeyValueLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
}

EagleBuffers::EagleBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1, "EAGLE does not support beam search");

    auto const maxNumSequences = maxBatchSize;

    auto const eagleModule = std::dynamic_pointer_cast<tensorrt_llm::runtime::EagleModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());

    auto const numPaths = eagleModule->getMaxNumPaths();
    auto const pathLen = eagleModule->getMaxPathLen();
    auto const maxDecodingDraftTokens = eagleModule->getMaxDecodingDraftTokens();

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;

    // input tensors
    engineInputs.temperatures = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);

    engineInputs.draftTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingDraftTokens}), TRTTokenIdType);
    engineInputs.draftLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineInputs.draftPaths
        = manager.gpu(ITensor::makeShape({maxNumSequences, numPaths, pathLen}), nvinfer1::DataType::kINT32);

    engineInputs.specDecodingGenerationLengths
        = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.specDecodingPositionOffsets
        = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.specDecodingPackedMasks = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);

    engineInputs.randomDataSample = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    engineInputs.randomDataValidation = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);

    engineInputs.eagleNetCtxRequestTypesHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetCtxContextLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetCtxPastKeyValueLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetGenRequestTypesHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetGenContextLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetGenPastKeyValueLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);

    // output tensors
    engineOutputs.nextDraftTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, numPaths, pathLen}), TRTTokenIdType);
    engineOutputs.nextDraftLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineOutputs.nextDraftPaths
        = manager.gpu(ITensor::makeShape({maxNumSequences, numPaths, pathLen}), nvinfer1::DataType::kINT32);

    engineOutputs.acceptedTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, pathLen}), nvinfer1::DataType::kINT32);
    engineOutputs.acceptedLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineOutputs.acceptedPaths = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);

    // helper tensors
    auto const& stream = manager.getStream();
    scanTempStorageBytes
        = tksd::invokeScanGenerationLengths(nullptr, 0, nullptr, nullptr, maxNumSequences, stream.get());
    reduceTempStorageBytes
        = tksd::invokeReduceMaxGenerationLengths(nullptr, 0, nullptr, nullptr, maxNumSequences, stream.get());
    scanReduceTempStorage = manager.gpu(std::max(reduceTempStorageBytes, scanTempStorageBytes));
    cumSumGenerationLengths = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    maxGenerationLength = manager.gpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    // pre-allocate empty tensors
    reshape(0, maxNumSequences, modelConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EagleBuffers::reshape(
    SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numSequences = numCtxSequences + numGenSequences;

    auto const eagleModule = std::dynamic_pointer_cast<tensorrt_llm::runtime::EagleModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());

    auto const maxDecodingTokens = eagleModule->getMaxDecodingTokens();

    // input tensors
    engineInputs.temperatures->reshape(ITensor::makeShape({numSequences}));

    auto draftTokensShape = engineInputs.draftTokens->getShape();
    draftTokensShape.d[0] = numSequences;
    engineInputs.draftTokens->reshape(draftTokensShape);
    auto draftLensShape = engineInputs.draftLens->getShape();
    draftLensShape.d[0] = numSequences;
    engineInputs.draftLens->reshape(draftLensShape);
    auto draftPathsShape = engineInputs.draftPaths->getShape();
    draftPathsShape.d[0] = numSequences;
    engineInputs.draftPaths->reshape(draftPathsShape);

    engineInputs.specDecodingGenerationLengths->reshape(ITensor::makeShape({numGenSequences}));
    engineInputs.specDecodingPositionOffsets->reshape(ITensor::makeShape({numGenSequences, maxDecodingTokens}));
    engineInputs.specDecodingPackedMasks->reshape(
        ITensor::makeShape({numGenSequences * maxDecodingTokens, common::ceilDiv(maxDecodingTokens, 32)}));

    engineInputs.randomDataSample->reshape(ITensor::makeShape({numSequences}));
    engineInputs.randomDataValidation->reshape(ITensor::makeShape({numSequences}));

    engineInputs.eagleNetCtxRequestTypesHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetCtxContextLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetCtxPastKeyValueLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetGenRequestTypesHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetGenContextLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetGenPastKeyValueLengthsHost->reshape(ITensor::makeShape({numSequences}));

    cumSumGenerationLengths->reshape(ITensor::makeShape({numSequences}));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleBuffers::setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 vocabSizePadded,
    ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers, ITensor const& contextPositionIds,
    runtime::EagleModule const& eagleModule, runtime::CudaStream const& stream) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    using runtime::bufferCast;

    tksd::PackEagleParams params;
    params.batchSize = numCtxSequences + numGenSequences;
    params.maxNumPaths = eagleModule.getMaxNumPaths();
    params.maxDecodingTokens = eagleModule.getMaxDecodingTokens();
    params.maxPathLength = eagleModule.getMaxPathLen();
    params.numContextRequests = numCtxSequences;
    params.numGenerationRequests = numGenSequences;

    params.batchSlots = bufferCast<SizeType32>(seqSlots);

    // Outputs from decoder -- inputs to the packing kernel
    params.inputTemperatures = bufferCast<float>(*draftBuffers.temperatures);
    params.inputRandomDataSample = bufferCast<float>(*draftBuffers.randomDataSample);
    params.inputRandomDataValidation = bufferCast<float>(*draftBuffers.randomDataValidation);

    params.inputNextDraftTokens = bufferCast<runtime::TokenIdType>(*draftBuffers.draftTokens);
    params.inputNextDraftLens = bufferCast<SizeType32>(*draftBuffers.draftLens);
    params.inputNextDraftPaths = bufferCast<SizeType32>(*draftBuffers.draftPaths);

    params.inputSpecDecodingGenerationLengths = bufferCast<SizeType32>(*draftBuffers.specDecodingGenerationLengths);
    params.inputSpecDecodingPositionOffsets = bufferCast<SizeType32>(*draftBuffers.specDecodingPositionOffsets);
    params.inputSpecDecodingPackedMasks = bufferCast<int32_t>(*draftBuffers.specDecodingPackedMasks);

    // Outputs of the packing kernel -- inputs to the engine
    params.outputTemperatures = bufferCast<float>(*engineInputs.temperatures);
    params.outputRandomDataSample = bufferCast<float>(*engineInputs.randomDataSample);
    params.outputRandomDataValidation = bufferCast<float>(*engineInputs.randomDataValidation);

    params.outputNextDraftTokens = bufferCast<runtime::TokenIdType>(*engineInputs.draftTokens);
    params.outputNextDraftLens = bufferCast<SizeType32>(*engineInputs.draftLens);
    params.outputNextDraftPaths = bufferCast<SizeType32>(*engineInputs.draftPaths);

    params.outputSpecDecodingGenerationLengths = bufferCast<SizeType32>(*engineInputs.specDecodingGenerationLengths);
    params.outputSpecDecodingPositionOffsets = bufferCast<SizeType32>(*engineInputs.specDecodingPositionOffsets);
    params.outputSpecDecodingPackedMasks = bufferCast<int32_t>(*engineInputs.specDecodingPackedMasks);

    params.maxGenerationLength = bufferCast<SizeType32>(*maxGenerationLength);
    params.cumSumGenerationLengths = bufferCast<SizeType32>(*cumSumGenerationLengths);

    params.checkParams();

    // Pack tensors from batch slot position to continuous array
    tksd::invokePackEagleGenerationLengths(params, stream.get());

    if (numGenSequences)
    {
        // Compute inclusive sum and max
        tksd::invokeScanReduceGenerationLengths(numGenSequences,
            bufferCast<SizeType32>(*engineInputs.specDecodingGenerationLengths),
            bufferCast<uint8_t>(*scanReduceTempStorage), scanTempStorageBytes,
            bufferCast<SizeType32>(*cumSumGenerationLengths), bufferCast<uint8_t>(*scanReduceTempStorage),
            reduceTempStorageBytes, bufferCast<SizeType32>(*maxGenerationLength), stream.get());
    }

    // Pack tensors from batch slot position to continuous array
    tksd::invokePackEagle(params, stream.get());

    // Pack host data.
    for (SizeType32 bi = 0; bi < params.batchSize; ++bi)
    {
        auto const batchSlot = params.batchSlots[bi];
        bufferCast<SizeType32>(*engineInputs.eagleNetCtxRequestTypesHost)[bi]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetCtxRequestTypesHost)[batchSlot];
        bufferCast<SizeType32>(*engineInputs.eagleNetCtxContextLengthsHost)[bi]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetCtxContextLengthsHost)[batchSlot];
        bufferCast<SizeType32>(*engineInputs.eagleNetCtxPastKeyValueLengthsHost)[bi]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetCtxPastKeyValueLengthsHost)[batchSlot];
        bufferCast<SizeType32>(*engineInputs.eagleNetGenRequestTypesHost)[bi]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetGenRequestTypesHost)[batchSlot];
        bufferCast<SizeType32>(*engineInputs.eagleNetGenContextLengthsHost)[bi]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetGenContextLengthsHost)[batchSlot];
        bufferCast<SizeType32>(*engineInputs.eagleNetGenPastKeyValueLengthsHost)[bi]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetGenPastKeyValueLengthsHost)[batchSlot];
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EagleBuffers::setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, ITensor const& requestTypes,
    ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers, ITensor const& contextPositionIds,
    runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& stream = runtime.getStream();

    auto const eagleModule
        = std::dynamic_pointer_cast<runtime::EagleModule const>(modelConfig.getSpeculativeDecodingModulePtr());

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const dtype = modelConfig.getDataType();

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        setFromInputs<float>(numCtxSequences, numGenSequences, vocabSizePadded, seqSlots, draftBuffers,
            contextPositionIds, *eagleModule, stream);
        break;
    case nvinfer1::DataType::kHALF:
        setFromInputs<half>(numCtxSequences, numGenSequences, vocabSizePadded, seqSlots, draftBuffers,
            contextPositionIds, *eagleModule, stream);
        break;
    default: TLLM_THROW("DataType %d not supported in EagleBuffers", static_cast<SizeType32>(dtype)); break;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EagleBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& /* worldConfig */) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // inputs
    inputBuffers.insert_or_assign("eagle_temperature", engineInputs.temperatures);

    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", engineInputs.specDecodingGenerationLengths);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", engineInputs.specDecodingPositionOffsets);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", engineInputs.specDecodingPackedMasks);

    inputBuffers.insert_or_assign("rand_data_sample", engineInputs.randomDataSample);
    inputBuffers.insert_or_assign("rand_data_validation", engineInputs.randomDataValidation);

    inputBuffers.insert_or_assign("draft_tokens", engineInputs.draftTokens);
    inputBuffers.insert_or_assign("draft_lens", engineInputs.draftLens);
    inputBuffers.insert_or_assign("draft_paths", engineInputs.draftPaths);

    inputBuffers.insert_or_assign("host_ctx_eagle_net_request_types", engineInputs.eagleNetCtxRequestTypesHost);
    inputBuffers.insert_or_assign("host_ctx_eagle_net_context_lengths", engineInputs.eagleNetCtxContextLengthsHost);
    inputBuffers.insert_or_assign(
        "host_ctx_eagle_net_past_key_value_lengths", engineInputs.eagleNetCtxPastKeyValueLengthsHost);
    inputBuffers.insert_or_assign("host_gen_eagle_net_request_types", engineInputs.eagleNetGenRequestTypesHost);
    inputBuffers.insert_or_assign("host_gen_eagle_net_context_lengths", engineInputs.eagleNetGenContextLengthsHost);
    inputBuffers.insert_or_assign(
        "host_gen_eagle_net_past_key_value_lengths", engineInputs.eagleNetGenPastKeyValueLengthsHost);

    // outputs
    outputBuffers.insert_or_assign("next_draft_tokens", engineOutputs.nextDraftTokens);
    outputBuffers.insert_or_assign("next_draft_lens", engineOutputs.nextDraftLens);

    outputBuffers.insert_or_assign("accepted_tokens", engineOutputs.acceptedTokens);
    outputBuffers.insert_or_assign("num_accepted_tokens", engineOutputs.acceptedLens);
    outputBuffers.insert_or_assign("accepted_paths", engineOutputs.acceptedPaths);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::runtime
