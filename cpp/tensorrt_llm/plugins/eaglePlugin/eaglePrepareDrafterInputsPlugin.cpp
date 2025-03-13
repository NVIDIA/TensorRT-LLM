/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "eaglePrepareDrafterInputsPlugin.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::EaglePrepareDrafterInputsPluginCreator;
using tensorrt_llm::plugins::EaglePrepareDrafterInputsPlugin;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::kernels::speculative_decoding;
using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

static char const* EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_VERSION{"1"};
static char const* EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_NAME{"EaglePrepareDrafterInputs"};
PluginFieldCollection EaglePrepareDrafterInputsPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EaglePrepareDrafterInputsPluginCreator::mPluginAttributes;

EaglePrepareDrafterInputsPlugin::EaglePrepareDrafterInputsPlugin(
    int32_t layerIdx, int32_t numLayers, int32_t maxNonLeavesPerLayer)
    : mLayerIdx(layerIdx)
    , mNumLayers(numLayers)
    , mMaxNonLeavesPerLayer(maxNonLeavesPerLayer)
{
}

void EaglePrepareDrafterInputsPlugin::initFieldsToSerialize()
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("layer_idx", &mLayerIdx, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("num_layers", &mNumLayers, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(
        PluginField("max_non_leaves_per_layer", &mMaxNonLeavesPerLayer, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
}

nvinfer1::IPluginCapability* EaglePrepareDrafterInputsPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    try
    {
        if (type == nvinfer1::PluginCapabilityType::kBUILD)
        {
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        }
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
        {
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        TLLM_CHECK(type == nvinfer1::PluginCapabilityType::kCORE);
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// IPluginV3 methods
nvinfer1::IPluginV3* EaglePrepareDrafterInputsPlugin::clone() noexcept
{
    auto clone = std::make_unique<EaglePrepareDrafterInputsPlugin>(*this);
    clone->initFieldsToSerialize();
    return clone.release();
}

// IPluginV3OneCore methods
char const* EaglePrepareDrafterInputsPlugin::getPluginName() const noexcept
{
    return EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_NAME;
}

char const* EaglePrepareDrafterInputsPlugin::getPluginVersion() const noexcept
{
    return EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_VERSION;
}

char const* EaglePrepareDrafterInputsPlugin::getPluginNamespace() const noexcept
{
    return tensorrt_llm::plugins::api::kDefaultNamespace;
}

// IPluginV3OneBuild methods
int32_t EaglePrepareDrafterInputsPlugin::getNbOutputs() const noexcept
{
    return 11;
}

int32_t EaglePrepareDrafterInputsPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

bool EaglePrepareDrafterInputsPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return (inOut[pos].desc.type == nvinfer1::DataType::kINT32) && (inOut[pos].desc.format == TensorFormat::kLINEAR);
}

int32_t EaglePrepareDrafterInputsPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[0] = nvinfer1::DataType::kINT32;
    outputTypes[1] = nvinfer1::DataType::kINT32;
    outputTypes[2] = nvinfer1::DataType::kINT32;
    outputTypes[3] = nvinfer1::DataType::kINT32;
    outputTypes[4] = nvinfer1::DataType::kINT32;
    outputTypes[5] = nvinfer1::DataType::kINT32;
    outputTypes[6] = nvinfer1::DataType::kINT32;
    outputTypes[7] = nvinfer1::DataType::kINT32;
    outputTypes[8] = nvinfer1::DataType::kINT32;
    outputTypes[9] = nvinfer1::DataType::kINT32;
    outputTypes[10] = nvinfer1::DataType::kINT32;
    outputTypes[11] = nvinfer1::DataType::kINT32;
    return 0;
}

int32_t EaglePrepareDrafterInputsPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(nbOutputs == 11);
    TLLM_CHECK(nbInputs == 15);
    TLLM_CHECK(nbShapeInputs == 0);
    auto const numTokens = inputs[getIdx(InputIdxEntry::INPUT_IDS)].d[0];
    auto const batchSizeExpr = inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)].d[0];
    auto const numGenRequestsExpr = inputs[getIdx(InputIdxEntry::SPEC_DECODING_GENERATION_LENGTHS)].d[0];
    auto const numInputGenTokensExpr = inputs[getIdx(InputIdxEntry::INPUT_GEN_TOKENS)].d[0];
    auto const maxDecodingLenExpr = inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)].d[1];
    auto const maxPathLenExpr = inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)].d[2];

    for (SizeType32 outputIndex = 0; outputIndex < nbOutputs; ++outputIndex)
    {
        if (outputIndex == getIdx(OutputIdxEntry::SEQUENCE_LENGTHS)
            || outputIndex == getIdx(OutputIdxEntry::CONTEXT_LENGTHS)
            || outputIndex == getIdx(OutputIdxEntry::SPEC_DECODING_GENERATION_LENGTHS))
        {
            outputs[outputIndex] = inputs[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)];
        }
        else if (outputIndex == getIdx(OutputIdxEntry::SPEC_DECODING_PACKED_MASK))
        {
            outputs[outputIndex].nbDims = 3;
            outputs[outputIndex].d[0] = batchSizeExpr;
            outputs[outputIndex].d[1] = maxDecodingLenExpr;
            outputs[outputIndex].d[2]
                = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *maxDecodingLenExpr, *exprBuilder.constant(32));
        }
        else if (outputIndex == getIdx(OutputIdxEntry::SPEC_DECODING_POSITION_OFFSETS))
        {
            outputs[outputIndex].nbDims = 2;
            outputs[outputIndex].d[0] = batchSizeExpr;
            outputs[outputIndex].d[1] = maxDecodingLenExpr;
        }
        else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_IDS)
            || outputIndex == getIdx(OutputIdxEntry::HIDDEN_STATES_INDICES)
            || (mLayerIdx == 0 && outputIndex == getIdx(OutputIdxEntry::POSITION_IDS)))
        {
            if (mLayerIdx == 0)
            {
                // We have at most numGenRequests * (mNumLayers + 1) accepted tokens per step for gen requests and
                // input_ids - numGenTokens tokens for context requests.
                auto numOutputGenTokensExpr = exprBuilder.operation(
                    DimensionOperation::kPROD, *numGenRequestsExpr, *exprBuilder.constant(mNumLayers + 1));
                auto numInputCtxTokensExpr
                    = exprBuilder.operation(DimensionOperation::kSUB, *numTokens, *numInputGenTokensExpr);
                outputs[outputIndex].nbDims = 1;
                outputs[outputIndex].d[0] = exprBuilder.operation(DimensionOperation::kMAX, *exprBuilder.constant(1),
                    *exprBuilder.operation(DimensionOperation::kSUM, *numOutputGenTokensExpr, *numInputCtxTokensExpr));
            }
            else
            {
                // At most we have mMaxNonLeavesPerLayer non-leaves at this layer.
                // And in total we pass all non-leaves + all their preceding nodes.
                // batchSize * mMaxNonLeavesPerLayer * layerIdx
                outputs[outputIndex].nbDims = 1;
                outputs[outputIndex].d[0] = exprBuilder.operation(DimensionOperation::kPROD,
                    *exprBuilder.operation(DimensionOperation::kPROD, *exprBuilder.constant(mLayerIdx),
                        *exprBuilder.constant(mMaxNonLeavesPerLayer)),
                    *batchSizeExpr);
            }
        }
        else if (mLayerIdx > 0 && outputIndex == getIdx(OutputIdxEntry::POSITION_IDS))
        {
            outputs[outputIndex].nbDims = 1;
            outputs[outputIndex].d[0] = batchSizeExpr;
        }
        else if (outputIndex == getIdx(OutputIdxEntry::LAST_TOKEN_INDICES))
        {
            outputs[outputIndex].nbDims = 1;
            outputs[outputIndex].d[0] = exprBuilder.operation(
                DimensionOperation::kPROD, *exprBuilder.constant(mMaxNonLeavesPerLayer), *batchSizeExpr);
        }
        else if (outputIndex == getIdx(OutputIdxEntry::NUM_LAST_TOKEN_INDICES))
        {
            outputs[outputIndex].nbDims = 1;
            outputs[outputIndex].d[0] = exprBuilder.constant(1);
        }
        else if (outputIndex == getIdx(OutputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS))
        {
            // batchSize * (maxPathLen - 1) + 1
            outputs[outputIndex].nbDims = 1;
            outputs[outputIndex].d[0] = exprBuilder.operation(DimensionOperation::kSUM, *exprBuilder.constant(1),
                *exprBuilder.operation(DimensionOperation::kPROD, *batchSizeExpr,
                    *exprBuilder.operation(DimensionOperation::kSUB, *maxPathLenExpr, *exprBuilder.constant(1))));
        }
    }
    return 0;
}

int32_t EaglePrepareDrafterInputsPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

nvinfer1::IPluginV3* EaglePrepareDrafterInputsPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* EaglePrepareDrafterInputsPlugin::getFieldsToSerialize() noexcept
{
    return &mFCToSerialize;
}

size_t EaglePrepareDrafterInputsPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    size_t workspaceSize{0};

    auto const batchSize = inputs[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].max.d[0];
    auto const maxDecodingTokens = inputs[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].max.d[1];

    if (mLayerIdx > 0)
    {
        SizeType32 constexpr NUM_BUFFERS{9};
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = batchSize * maxDecodingTokens * sizeof(int8_t);                     // isLeafMask
        workspaces[1] = batchSize * maxDecodingTokens * sizeof(SizeType32);                 // selectedDraftIndices
        workspaces[2] = batchSize * maxDecodingTokens * sizeof(SizeType32);                 // selectedDraftPosOffsets
        workspaces[3] = batchSize * sizeof(SizeType32);                                     // numSelectedDraftIndices
        workspaces[4] = batchSize * maxDecodingTokens * maxDecodingTokens * sizeof(int8_t); // selectedMasks
        workspaces[5] = (batchSize + 1) * sizeof(SizeType32);                               // cumSumGenerationLengths
        workspaces[6] = batchSize * maxDecodingTokens * sizeof(SizeType32);                 // nonLeavesInLevelOffsets
        workspaces[7] = batchSize * maxDecodingTokens * sizeof(SizeType32); // parentNonLeafInLevelOffset
        workspaces[8] = 1 * sizeof(SizeType32);                             // maxGenerationLength
        workspaceSize = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    }

    return workspaceSize;
}

void EaglePrepareDrafterInputsPlugin::prepareCtxEagleNetData(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputDesc[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)].dims.d[0];

    auto const numTokens = inputDesc[getIdx(InputIdxEntry::INPUT_IDS)].dims.d[0];
    auto const numGenRequests = inputDesc[getIdx(InputIdxEntry::SPEC_DECODING_GENERATION_LENGTHS)].dims.d[0];
    auto const numInputGenTokens = inputDesc[getIdx(InputIdxEntry::INPUT_GEN_TOKENS)].dims.d[0];

    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::ACCEPTED_TOKENS)].dims.d[1];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].dims.d[1];

    auto eagleNetSequenceLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::SEQUENCE_LENGTHS)]);
    auto eagleNetContextLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::CONTEXT_LENGTHS)]);
    auto outputIds = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_IDS)]);
    auto positionIds = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::POSITION_IDS)]);
    auto hiddenStatesIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::HIDDEN_STATES_INDICES)]);
    auto lastTokenIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::LAST_TOKEN_INDICES)]);
    auto numLastTokenIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::NUM_LAST_TOKEN_INDICES)]);
    auto hiddenSizeBatchLevelStarts
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS)]);

    auto inputIds = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_IDS)]);
    auto chunkedContextNextTokens
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::CHUNKED_CONTEXT_NEXT_TOKENS)]);
    auto baseNetSequenceLengths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)]);
    auto baseNetContextLengths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::CONTEXT_LENGTHS)]);
    auto acceptedTokens = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::ACCEPTED_TOKENS)]);
    auto acceptedLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::ACCEPTED_LENS)]);
    auto prevDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PREV_DRAFT_LENS)]);
    auto prevPaths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)]);
    auto bestPathIds = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::ACCEPTED_PATHS)]);

    auto const numOutputTokens = (numTokens - numInputGenTokens) + (numGenRequests * (mNumLayers + 1));
    cudaMemsetAsync(positionIds, 0, numOutputTokens * sizeof(SizeType32), stream);
    cudaMemsetAsync(hiddenStatesIndices, 0, numOutputTokens * sizeof(SizeType32), stream);

    invokePrepareCtxEagleNetInputs(eagleNetSequenceLengths, eagleNetContextLengths, outputIds, positionIds,
        hiddenStatesIndices, lastTokenIndices, numLastTokenIndices, hiddenSizeBatchLevelStarts, inputIds,
        chunkedContextNextTokens, baseNetSequenceLengths, baseNetContextLengths, acceptedTokens, acceptedLens,
        prevDraftLens, prevPaths, bestPathIds, batchSize, maxPathLen, maxDecodingTokens, mMaxNonLeavesPerLayer, stream);

    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EaglePrepareDrafterInputsPlugin::prepareGenEagleNetData(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputDesc[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].dims.d[1];
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].dims.d[2];

    auto eagleNetSequenceLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::SEQUENCE_LENGTHS)]);
    auto eagleNetContextLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::CONTEXT_LENGTHS)]);
    auto outputIds = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_IDS)]);
    auto positionIds = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::POSITION_IDS)]);
    auto specDecodingGenLengths
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::SPEC_DECODING_GENERATION_LENGTHS)]);
    auto specDecodingPositionOffsets
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::SPEC_DECODING_POSITION_OFFSETS)]);
    auto specDecodingPackedMasks
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::SPEC_DECODING_PACKED_MASK)]);
    auto hiddenStatesIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::HIDDEN_STATES_INDICES)]);
    auto lastTokenIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::LAST_TOKEN_INDICES)]);
    auto numLastTokenIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::NUM_LAST_TOKEN_INDICES)]);
    auto outputHiddenSizeBatchStartsPerLevel
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS)]);

    auto eagleNet0SequenceLengths
        = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)]);
    auto eagleNet0ContextLength = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::CONTEXT_LENGTHS)]);
    auto nextDraftPaths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)]);
    auto nextDraftIds = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::NEXT_DRAFT_TOKENS)]);
    auto inputHiddenSizeBatchStartsPerLevel
        = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS)]);

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};

    int8_t* isLeafMask = reinterpret_cast<int8_t*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(int8_t)));
    TokenIdType* selectedDraftIndices = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(TokenIdType)));
    SizeType32* selectedDraftPosOffsets = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(SizeType32)));
    SizeType32* numSelectedDraftIndices
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(SizeType32)));
    bool* selectedMasks = reinterpret_cast<bool*>(tc::nextWorkspacePtr(
        workspaceBytePtr, offset, batchSize * maxDecodingTokens * maxDecodingTokens * sizeof(int8_t)));
    SizeType32* cumSumGenerationLengths = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, (batchSize + 1) * sizeof(SizeType32)));
    SizeType32* nonLeavesInLevelOffsets = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(SizeType32)));
    SizeType32* parentNonLeafInLevelOffset = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(SizeType32)));
    SizeType32* maxGenerationLength
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, 1 * sizeof(SizeType32)));

    cudaMemsetAsync(hiddenStatesIndices, 0, batchSize * mMaxNonLeavesPerLayer * mLayerIdx * sizeof(SizeType32), stream);
    cudaMemsetAsync(selectedMasks, 0, batchSize * maxDecodingTokens * maxDecodingTokens * sizeof(int8_t), stream);
    // Prefill mask setting all to leaves.
    cudaMemsetAsync(isLeafMask, 1, batchSize * maxDecodingTokens * sizeof(int8_t), stream);

    PrepareGenEagleNetInputsParams params;
    params.nextSequenceLengths = eagleNetSequenceLengths;
    params.nextContextLengths = eagleNetContextLengths;
    params.outputIds = outputIds;
    params.positionIds = positionIds;
    params.specDecodingGenLengths = specDecodingGenLengths;
    params.specDecodingPositionOffsets = specDecodingPositionOffsets;
    params.specDecodingPackedMasks = specDecodingPackedMasks;
    params.hiddenStatesIndices = hiddenStatesIndices;
    params.lastTokenIndices = lastTokenIndices;
    params.numLastTokenIndices = numLastTokenIndices;
    params.outputHiddenSizeBatchStartsPerLevel = outputHiddenSizeBatchStartsPerLevel;

    // tmp data
    params.isLeafMask = isLeafMask;
    params.selectedDraftIndices = selectedDraftIndices;
    params.selectedDraftPosOffsets = selectedDraftPosOffsets;
    params.numSelectedDraftIndices = numSelectedDraftIndices;
    params.selectedMasks = selectedMasks;
    params.cumSumGenerationLengths = cumSumGenerationLengths;
    params.maxGenerationLength = maxGenerationLength;
    params.nonLeavesInLevelOffsets = nonLeavesInLevelOffsets;
    params.parentNonLeafInLevelOffset = parentNonLeafInLevelOffset;

    params.nextDraftIds = nextDraftIds;
    params.eagleNet0SequenceLengths = eagleNet0SequenceLengths;
    params.prevContextLengths = eagleNet0ContextLength;
    params.nextPaths = nextDraftPaths;
    params.inputHiddenSizeBatchStartsPerLevel = inputHiddenSizeBatchStartsPerLevel;
    params.levelIdx = mLayerIdx;
    params.batchSize = batchSize;
    params.maxPathLen = maxPathLen;
    params.maxDecodingTokens = maxDecodingTokens;
    params.maxNonLeavesPerLayer = mMaxNonLeavesPerLayer;
    params.stream = stream;

    params.checkParams();

    invokePrepareGenEagleNetInputs(params);

    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

int EaglePrepareDrafterInputsPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // First EagleNet instance (EagleNet0) is always chunked context attn,
    // where we process either context tokens or newly accepted tokens and append them to EagleNet KV cache.

    // For all following EagleNetX (X > 0) instances there is need for masked spec decoding attn.
    // Ideally with mask for context.
    // Let's say we have prompt ABCD and two variants of tokens spec decoding tokens E and F
    // predicted by EagleNet0. If we draw full attn mask, it becomes:
    //  |A|B|C|D|E|F
    // E|1|1|1|1|1|0
    // F|1|1|1|1|0|1
    //
    // In the next step we predict token G from ABCDE branch and token H from ABCDF branch -- like beam search.
    // And we'd need spec decoding mask that includes kv cache:
    //  |A|B|C|D|E|F|G|H
    // G|1|1|1|1|1|0|1|0
    // H|1|1|1|1|0|1|0|1
    //
    // But TRT-LLM does not support such mask for now. We can only provide
    //  |G|H
    // G|1|0
    // H|0|1
    // , which is wrong mask.
    //
    // For now we WAR this by passing EFGH for the EagleNet1 with right mask
    // and using only G and H logits for sampling, but that's redundant compute:
    //  |E|F|G|H
    // E|1|0|0|0
    // F|0|1|0|0
    // G|1|0|1|0
    // H|0|1|0|1

    if (mLayerIdx == 0)
    {
        prepareCtxEagleNetData(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        prepareGenEagleNetData(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    return 0;
}

///////////////

EaglePrepareDrafterInputsPluginCreator::EaglePrepareDrafterInputsPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_layers", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("max_non_leaves_per_layer", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EaglePrepareDrafterInputsPluginCreator::getPluginName() const noexcept
{
    return EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_NAME;
}

char const* EaglePrepareDrafterInputsPluginCreator::getPluginVersion() const noexcept
{
    return EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_VERSION;
}

PluginFieldCollection const* EaglePrepareDrafterInputsPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* EaglePrepareDrafterInputsPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept
{
    try
    {
        int32_t layerIdx{0};
        int32_t numLayers{0};
        int32_t maxNonLeavesPerLayer{0};
        // Read configurations from each fields
        for (int i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fc->fields[i].name;
            if (!strcmp(attrName, "layer_idx"))
            {
                TLLM_CHECK(fc->fields[i].type == PluginFieldType::kINT32);
                layerIdx = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (!strcmp(attrName, "num_layers"))
            {
                TLLM_CHECK(fc->fields[i].type == PluginFieldType::kINT32);
                numLayers = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (!strcmp(attrName, "max_non_leaves_per_layer"))
            {
                TLLM_CHECK(fc->fields[i].type == PluginFieldType::kINT32);
                maxNonLeavesPerLayer = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        return new EaglePrepareDrafterInputsPlugin(layerIdx, numLayers, maxNonLeavesPerLayer);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* EaglePrepareDrafterInputsPluginCreator::getPluginNamespace() const noexcept
{
    return tensorrt_llm::plugins::api::kDefaultNamespace;
}
