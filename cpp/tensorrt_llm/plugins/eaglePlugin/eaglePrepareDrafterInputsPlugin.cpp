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

EaglePrepareDrafterInputsPlugin::EaglePrepareDrafterInputsPlugin(int32_t layerIdx)
    : mLayerIdx(layerIdx)
{
}

// Parameterized constructor
EaglePrepareDrafterInputsPlugin::EaglePrepareDrafterInputsPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mLayerIdx);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        static_cast<int>(length), static_cast<int>(d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* EaglePrepareDrafterInputsPlugin::clone() const noexcept
{
    auto* plugin = new EaglePrepareDrafterInputsPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs EaglePrepareDrafterInputsPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex < 12);
    TLLM_CHECK(nbInputs == 12);
    auto const numTokens = inputs[getIdx(InputIdxEntry::INPUT_IDS)].d[0];
    auto const batchSizeExpr = inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)].d[0];
    auto const maxDecodingLenExpr = inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)].d[1];
    auto const maxPathLenExpr = inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)].d[2];

    nvinfer1::DimsExprs ret;
    if (outputIndex == getIdx(OutputIdxEntry::SEQUENCE_LENGTHS)
        || outputIndex == getIdx(OutputIdxEntry::CONTEXT_LENGTHS)
        || outputIndex == getIdx(OutputIdxEntry::SPEC_DECODING_GENERATION_LENGTHS))
    {
        ret = inputs[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)];
    }
    else if (outputIndex == getIdx(OutputIdxEntry::SPEC_DECODING_PACKED_MASK))
    {
        // FIXME
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingLenExpr;
        ret.d[2] = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *maxDecodingLenExpr, *exprBuilder.constant(32));
    }
    else if (outputIndex == getIdx(OutputIdxEntry::SPEC_DECODING_POSITION_OFFSETS))
    {
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingLenExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_IDS) || outputIndex == getIdx(OutputIdxEntry::POSITION_IDS)
        || outputIndex == getIdx(OutputIdxEntry::HIDDEN_STATES_INDICES))
    {
        ret.nbDims = 1;
        ret.d[0] = exprBuilder.operation(DimensionOperation::kSUM,
            *exprBuilder.operation(DimensionOperation::kPROD, *maxDecodingLenExpr, *batchSizeExpr), *numTokens);
    }
    else if (outputIndex == getIdx(OutputIdxEntry::LAST_TOKEN_INDICES))
    {
        ret.nbDims = 1;
        ret.d[0] = exprBuilder.operation(DimensionOperation::kPROD, *maxDecodingLenExpr, *batchSizeExpr);
    }
    else if (outputIndex == getIdx(OutputIdxEntry::NUM_OUTPUT_TOKENS)
        || outputIndex == getIdx(OutputIdxEntry::NUM_LAST_TOKEN_INDICES))
    {
        ret.nbDims = 1;
        ret.d[0] = exprBuilder.constant(1);
    }
    else if (outputIndex == getIdx(OutputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS))
    {
        // batchSize * (maxPathLen - 1) + 1
        ret.nbDims = 1;
        ret.d[0] = exprBuilder.operation(DimensionOperation::kSUM, *exprBuilder.constant(1),
            *exprBuilder.operation(DimensionOperation::kPROD, *batchSizeExpr,
                *exprBuilder.operation(DimensionOperation::kSUB, *maxPathLenExpr, *exprBuilder.constant(1))));
    }
    return ret;
}

bool EaglePrepareDrafterInputsPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void EaglePrepareDrafterInputsPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t EaglePrepareDrafterInputsPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    size_t workspaceSize{0};

    auto const batchSize = inputs[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)].dims.d[0];
    auto const maxDecodingTokens = inputs[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].dims.d[1];

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
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::ACCEPTED_TOKENS)].dims.d[1];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::NEXT_DRAFT_PATHS)].dims.d[1];

    auto eagleNetSequenceLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::SEQUENCE_LENGTHS)]);
    auto eagleNetContextLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::CONTEXT_LENGTHS)]);
    auto outputIds = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_IDS)]);
    auto positionIds = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::POSITION_IDS)]);
    auto hiddenStatesIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::HIDDEN_STATES_INDICES)]);
    auto lastTokenIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::LAST_TOKEN_INDICES)]);
    auto numOutputTokens = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::NUM_OUTPUT_TOKENS)]);
    auto numLastTokenIndices = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::NUM_LAST_TOKEN_INDICES)]);
    auto hiddenSizeBatchLevelStarts
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS)]);

    auto inputIds = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_IDS)]);
    auto baseNetSequenceLengths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)]);
    auto baseNetContextLengths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::CONTEXT_LENGTHS)]);
    auto acceptedTokens = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::ACCEPTED_TOKENS)]);
    auto acceptedLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::ACCEPTED_LENS)]);
    auto prevDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PREV_DRAFT_LENS)]);
    auto prevPaths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PREV_DRAFT_PATHS)]);
    auto bestPathIds = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::ACCEPTED_PATHS)]);

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};

    invokePrepareCtxEagleNetInputs(eagleNetSequenceLengths, eagleNetContextLengths, outputIds, positionIds,
        hiddenStatesIndices, lastTokenIndices, numOutputTokens, numLastTokenIndices, hiddenSizeBatchLevelStarts,
        inputIds, baseNetSequenceLengths, baseNetContextLengths, acceptedTokens, acceptedLens, prevDraftLens, prevPaths,
        bestPathIds, batchSize, maxPathLen, maxDecodingTokens, stream);

    sync_check_cuda_error();

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
    auto numOutputTokens = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::NUM_OUTPUT_TOKENS)]);
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
    params.numOutputTokens = numOutputTokens;
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
    params.stream = stream;

    params.checkParams();

    invokePrepareGenEagleNetInputs(params);

    sync_check_cuda_error();

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

// IPluginV2Ext Methods
nvinfer1::DataType EaglePrepareDrafterInputsPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[getIdx(InputIdxEntry::SEQUENCE_LENGTHS)];
}

// IPluginV2 Methods

char const* EaglePrepareDrafterInputsPlugin::getPluginType() const noexcept
{
    return EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_NAME;
}

char const* EaglePrepareDrafterInputsPlugin::getPluginVersion() const noexcept
{
    return EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_VERSION;
}

int EaglePrepareDrafterInputsPlugin::getNbOutputs() const noexcept
{
    return 12;
}

int EaglePrepareDrafterInputsPlugin::initialize() noexcept
{
    return 0;
}

void EaglePrepareDrafterInputsPlugin::terminate() noexcept {}

size_t EaglePrepareDrafterInputsPlugin::getSerializationSize() const noexcept
{
    return sizeof(mLayerIdx);
}

void EaglePrepareDrafterInputsPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mLayerIdx);
    assert(d == a + getSerializationSize());
}

void EaglePrepareDrafterInputsPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

EaglePrepareDrafterInputsPluginCreator::EaglePrepareDrafterInputsPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32, 0));
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

IPluginV2* EaglePrepareDrafterInputsPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int32_t layerIdx;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "layer_idx"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            layerIdx = *static_cast<int32_t const*>(fields[i].data);
        }
    }

    try
    {
        auto* obj = new EaglePrepareDrafterInputsPlugin(layerIdx);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EaglePrepareDrafterInputsPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call EaglePrepareDrafterInputsPlugin::destroy()
    try
    {
        auto* obj = new EaglePrepareDrafterInputsPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
