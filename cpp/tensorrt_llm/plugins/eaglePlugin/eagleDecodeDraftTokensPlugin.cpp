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
#include "eagleDecodeDraftTokensPlugin.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/medusaDecodingKernels.h"
#include "tensorrt_llm/runtime/common.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::EagleDecodeDraftTokensPluginCreator;
using tensorrt_llm::plugins::EagleDecodeDraftTokensPlugin;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::kernels::speculative_decoding;
using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

static char const* EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION{"1"};
static char const* EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME{"EagleDecodeDraftTokens"};
PluginFieldCollection EagleDecodeDraftTokensPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EagleDecodeDraftTokensPluginCreator::mPluginAttributes;

EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(nvinfer1::DataType type, int32_t layerIdx, bool topKSampling)
    : mDtype(type)
    , mLayerIdx(layerIdx)
    , mTopKSampling(topKSampling)
{
    TLLM_CHECK_WITH_INFO(mTopKSampling, "Multinomial sampling is not supported yet.");
}

// Parameterized constructor
EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDtype);
    read(d, mLayerIdx);
    read(d, mTopKSampling);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        static_cast<int>(length), static_cast<int>(d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* EagleDecodeDraftTokensPlugin::clone() const noexcept
{
    auto* plugin = new EagleDecodeDraftTokensPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs EagleDecodeDraftTokensPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex < 2);
    TLLM_CHECK(nbInputs == 6);
    auto const batchSizeExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[0];
    auto const maxDecodingTokensExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[1];
    auto const maxDecodingDraftTokensExpr
        = exprBuilder.operation(DimensionOperation::kSUB, *maxDecodingTokensExpr, *exprBuilder.constant(1));

    nvinfer1::DimsExprs ret;
    if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_DRAFT_TOKEN_IDS))
    {
        // output_draft_token_ids
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_DRAFT_LENS))
    {
        // output_draft_lens
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "Wrong outputIndex %d in EagleDecodeDraftTokensPlugin::getOutputDimensions", outputIndex);
    }
    return ret;
}

bool EagleDecodeDraftTokensPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(nbInputs == 6 && nbOutputs == 2);
    TLLM_CHECK(pos < nbInputs + nbOutputs);

    if (pos == getIdx(InputIdxEntry::LOGITS)) // logits (input)
    {
        return (inOut[pos].type == mDtype) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == getIdx(InputIdxEntry::RAND_SAMPLE)) // randSample (input)
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else // path (input) or draft_token_ids (input/output) or draft_lens (input/output)
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void EagleDecodeDraftTokensPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

template <typename T>
size_t EagleDecodeDraftTokensPlugin::getWorkspaceSizeType(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    size_t workspaceSize{0};
    auto const numInputLogits = inputs[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const batchSize = inputs[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const vocabSizePadded = inputs[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const maxDecodingTokens = inputs[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    auto const maxTopK = std::min(TOP_K_MAX, SizeType32(maxDecodingDraftTokens));

    // Greedy sampling
    if (mTopKSampling)
    {
        // 0. TopK sampling workspace
        auto const draftTokenSamplingWorkspaceSize
            = getTopKWorkspaceSize<T>(numInputLogits, /* maxTokensPerStep */ 1, /* maxTopK */ maxTopK, vocabSizePadded);

        // 1. TopKs [numInputLogits]
        auto const topKsSize = numInputLogits * sizeof(SizeType32);

        // 2. Topks offset [batchSize]
        // Each request will have different number of logits that need to be sampled
        // This tensor will record the start offset of the topK for each request
        auto const topKOffsetSize = batchSize * sizeof(SizeType32);

        // 3. Logits ptrs [numInputLogits]
        auto const logitsPtrsSize = numInputLogits * sizeof(T*);

        // 4. Output ids ptrs [numInputLogits][maxDecodingDraftTokens]
        auto const outputIdsPtrsSize = numInputLogits * sizeof(TokenIdType*);

        // 5. Output ids (temporary buffer) [numInputLogits * maxDecodingDraftTokens]
        auto const outputIdsSize = numInputLogits * maxDecodingDraftTokens * sizeof(TokenIdType);

        // 6. Number of successors for each nodes, extract from the paths and layerId
        // [batchSize * maxDecodingTokens]
        auto const numSuccessorsForEachNodeSize = batchSize * maxDecodingTokens * sizeof(SizeType32);

        // 7. Flag whether to do decoding or not. SamplingTopK is done for numInputLogits tokens.
        // But only sum(numValidLogitsPerRequest[:]) of them are valid.
        // [batchSize * maxDecodingTokens]
        auto const skipDecodeSize = numInputLogits * sizeof(bool);

        SizeType32 constexpr NUM_BUFFERS{8};
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = draftTokenSamplingWorkspaceSize;
        workspaces[1] = topKsSize;
        workspaces[2] = topKOffsetSize;
        workspaces[3] = logitsPtrsSize;
        workspaces[4] = outputIdsPtrsSize;
        workspaces[5] = outputIdsSize;
        workspaces[6] = numSuccessorsForEachNodeSize;
        workspaces[7] = skipDecodeSize;
        workspaceSize = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    }
    else
    {
        // TODO fill me
        // Multinomial sampling
        TLLM_CHECK_WITH_INFO(false, "Multinomial sampling is not supported yet.");
    }

    return workspaceSize;
}

size_t EagleDecodeDraftTokensPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    auto const logitsType = inputs[getIdx(InputIdxEntry::LOGITS)].type;
    if (logitsType == nvinfer1::DataType::kFLOAT)
    {
        return getWorkspaceSizeType<float>(inputs, nbInputs, outputs, nbOutputs);
    }
    else if (logitsType == nvinfer1::DataType::kHALF)
    {
        return getWorkspaceSizeType<__half>(inputs, nbInputs, outputs, nbOutputs);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported logits type");
    }
    return 0;
}

template <typename T>
void EagleDecodeDraftTokensPlugin::doTopKSampling(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numInputLogits = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const vocabSizePadded = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const batchSize = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[2];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    auto const maxTopK = std::min(TOP_K_MAX, SizeType32(maxDecodingDraftTokens));

    // Plugin inputs
    auto logits = static_cast<T const*>(inputs[getIdx(InputIdxEntry::LOGITS)]);
    auto numValidLogits = static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::NUM_VALID_LOGITS)]);
    auto randSample = static_cast<float const*>(inputs[getIdx(InputIdxEntry::RAND_SAMPLE)]);
    auto paths = static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PATHS)]);
    auto pluginInputDraftTokenIdsPtrs
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_DRAFT_TOKEN_IDS)]);
    auto pluginInputDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::INPUT_DRAFT_LENS)]);

    // Plugin outputs
    auto pluginOutputDraftTokenIdsPtrs
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_DRAFT_TOKEN_IDS)]);
    auto pluginOutputDraftLens = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_DRAFT_LENS)]);

    // Get workspace
    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};

    // Workspace 0: Sampling workspace.
    // Treat numInputLogits as batchSize
    auto const samplingWorkspaceSize
        = getTopKWorkspaceSize<T>(numInputLogits, /* maxTokensPerStep */ 1, /* maxTopK */ maxTopK, vocabSizePadded);

    void* workspaceSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, samplingWorkspaceSize));

    // Workspace 1: Topks tensor: shape [numInputLogits]
    SizeType32* topKs = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(SizeType32)));

    // Workspace 2: topKOffset tensor: shape: [batchSize], number of nodes that have successors for each requests
    SizeType32* topKOffset
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(SizeType32)));

    // Workspace 3: logits pointers tensor: shape: [numInputLogits][1, vocabSizePadded], each logits has its own logits
    // buffer
    T const** logitsPtrs
        = reinterpret_cast<T const**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(T*)));

    // Workspace 4: outputIds pointers tensor: shape [numInputLogits][maxDecodingDraftTokens]
    TokenIdType** outputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(TokenIdType*)));

    // Workspace 5: outputIds tensor: flatten outputIds, shape [numInputLogits * maxDecodingDraftTokens]
    TokenIdType* outputIdsFlatten = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * maxDecodingDraftTokens * sizeof(TokenIdType)));

    // Workspace 6: number of successors for each nodes tensor: shape [batchSize * maxDecodingTokens]
    SizeType32* numSuccessorsForEachNode = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(SizeType32)));

    // Workspace 7: skip decoding mask [numInputLogits]
    bool* skipDecode
        = reinterpret_cast<bool*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(bool)));

    // Fill logitsPtrs from logits, fill outputIdsPtrs from outputIdsFlatten and fill decodingTokens
    invokeAssembleDraftLogitsOffsets(logitsPtrs, logits, outputIdsPtrs, outputIdsFlatten, skipDecode, numValidLogits,
        numInputLogits, batchSize, maxDecodingDraftTokens, vocabSizePadded, stream);
    sync_check_cuda_error();

    invokeExtractTopKsFromPath(paths, topKs, topKOffset, numSuccessorsForEachNode, mLayerIdx, batchSize,
        maxDecodingTokens, maxPathLen, stream);
    sync_check_cuda_error();

    TopKSamplingKernelParams<T> params{};
    params.logProbsPtrs = logitsPtrs;
    params.outputIdsPtrs = outputIdsPtrs;
    params.workspace = workspaceSampling;
    params.maxTopK = maxTopK;
    params.topKs = topKs;
    params.batchSize = numInputLogits;
    params.maxBatchSize = numInputLogits;
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = vocabSizePadded;
    params.returnAllSelectedTokens = true;
    params.skipDecode = skipDecode;

    invokeBatchTopKSampling(params, stream);
    sync_check_cuda_error();

    // Copy output token id from outputIdsPtrs to the plugin output buffer
    invokeCopyOutputTokensIds(outputIdsPtrs, topKs, topKOffset, pluginInputDraftTokenIdsPtrs, pluginInputDraftLens,
        numValidLogits, pluginOutputDraftTokenIdsPtrs, pluginOutputDraftLens, mLayerIdx, batchSize,
        maxDecodingDraftTokens, stream);
    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodeDraftTokensPlugin::enqueueType(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // TODO split batch into greedy and non-greedy and execute both paths
    if (mTopKSampling)
    {
        doTopKSampling<T>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        // TODO fill me
        TLLM_CHECK_WITH_INFO(false, "Multinomial sampling is not supported yet");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

int EagleDecodeDraftTokensPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    auto const logitsType = inputDesc[getIdx(InputIdxEntry::LOGITS)].type;
    if (logitsType == nvinfer1::DataType::kFLOAT)
    {
        enqueueType<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (logitsType == nvinfer1::DataType::kHALF)
    {
        enqueueType<__half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported logits type");
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType EagleDecodeDraftTokensPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index < 2);
    // Both draft_token_ids and draft_lens are int32 type, same as path
    return inputTypes[getIdx(InputIdxEntry::PATHS)];
}

// IPluginV2 Methods

char const* EagleDecodeDraftTokensPlugin::getPluginType() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME;
}

char const* EagleDecodeDraftTokensPlugin::getPluginVersion() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION;
}

int EagleDecodeDraftTokensPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int EagleDecodeDraftTokensPlugin::initialize() noexcept
{
    return 0;
}

void EagleDecodeDraftTokensPlugin::terminate() noexcept {}

size_t EagleDecodeDraftTokensPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDtype) + sizeof(mLayerIdx) + sizeof(mTopKSampling);
}

void EagleDecodeDraftTokensPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDtype);
    write(d, mLayerIdx);
    write(d, mTopKSampling);
    TLLM_CHECK(d == a + getSerializationSize());
}

void EagleDecodeDraftTokensPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

EagleDecodeDraftTokensPluginCreator::EagleDecodeDraftTokensPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("top_k_sampling", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EagleDecodeDraftTokensPluginCreator::getPluginName() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME;
}

char const* EagleDecodeDraftTokensPluginCreator::getPluginVersion() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION;
}

PluginFieldCollection const* EagleDecodeDraftTokensPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* EagleDecodeDraftTokensPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int32_t layerIdx;
    nvinfer1::DataType type;
    bool topKSampling;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "layer_idx"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            layerIdx = *static_cast<int32_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "top_k_sampling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            topKSampling = static_cast<bool>(*static_cast<int32_t const*>(fields[i].data));
        }
    }

    try
    {
        auto* obj = new EagleDecodeDraftTokensPlugin(type, layerIdx, topKSampling);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EagleDecodeDraftTokensPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call EagleDecodeDraftTokensPlugin::destroy()
    try
    {
        auto* obj = new EagleDecodeDraftTokensPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
