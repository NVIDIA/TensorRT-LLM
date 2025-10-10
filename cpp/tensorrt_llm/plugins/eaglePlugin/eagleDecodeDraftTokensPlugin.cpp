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
#include "tensorrt_llm/runtime/iTensor.h"

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

EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(
    nvinfer1::DataType type, int32_t layerIdx, int32_t numEagleLayers, bool topKSampling)
    : mDtype(type)
    , mLayerIdx(layerIdx)
    , mNumEagleLayers(numEagleLayers)
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
    read(d, mNumEagleLayers);
    read(d, mTopKSampling);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
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
    TLLM_CHECK(outputIndex < getNbOutputs());
    TLLM_CHECK(nbInputs == 12);
    auto const batchSizeExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[0];
    auto const maxDecodingTokensExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[1];
    auto const maxPathLengthExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[2];
    auto const maxDecodingDraftTokensExpr
        = exprBuilder.operation(DimensionOperation::kSUB, *maxDecodingTokensExpr, *exprBuilder.constant(1));

    auto const numEagleLayersExpr
        = exprBuilder.operation(DimensionOperation::kSUB, *maxPathLengthExpr, *exprBuilder.constant(1));
    auto const maxDecodingDraftTokensSquareExpr
        = exprBuilder.operation(DimensionOperation::kPROD, *maxDecodingDraftTokensExpr,
            *maxDecodingDraftTokensExpr); // maxDecodingDraftTokensExpr * maxDecodingDraftTokensExpr

    nvinfer1::DimsExprs ret;
    if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_DRAFT_TOKEN_IDS))
    {
        // output_draft_token_ids: [batch_size, max_decoding_draft_tokens]
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_DRAFT_LENS))
    {
        // output_draft_lens: [batch_size]
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_PATHS))
    {
        // output_path: [batch_size, max_decoding_tokens, max_path_len]
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingTokensExpr;
        ret.d[2] = maxPathLengthExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES))
    {
        // output_current_scores: [batch_size, max_decoding_draft_tokens]
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_NEXT_EXPAND_INDICES))
    {
        // output_next_expand_index
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES))
    {
        // output_all_layers_scores:
        // [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = numEagleLayersExpr;
        ret.d[2] = maxDecodingDraftTokensSquareExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS))
    {
        // output_all_layers_draft_token_ids:
        // [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = numEagleLayersExpr;
        ret.d[2] = maxDecodingDraftTokensSquareExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR))
    {
        // output_all_layers_draft_token_ids_predecessor
        // [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = numEagleLayersExpr;
        ret.d[2] = maxDecodingDraftTokensSquareExpr;
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
    TLLM_CHECK(nbInputs == 12 && nbOutputs == getNbOutputs());
    TLLM_CHECK(pos < nbInputs + nbOutputs);

    if (pos == getIdx(InputIdxEntry::LOGITS))
    {
        // input: logits
        // output: output_all_layers_scores
        return (inOut[pos].type == mDtype) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES) || pos == getIdx(InputIdxEntry::INPUT_PREV_SCORES)
        || pos == nbInputs + getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES)
        || pos == nbInputs + getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES))
    {
        // input: rand_sample, input_all_layers_scores, input_prev_scores
        // output: output_all_layers_scores, output_current_scores
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        // input: path, num_valid_logits, use_dynamic_tree, dynamic_tree_max_topK, input_draft_token_ids,
        //        input_draft_lens, input_current_expand_index, input_all_layers_draft_token_ids
        // output: output_draft_token_ids, output_draft_lens, output_path, output_next_expand_index
        //        output_all_layers_draft_token_ids, output_all_alyers_draft_token_predecessor
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
    auto const maxTopK = maxDecodingDraftTokens;
    auto const mNumEagleLayers = inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)].dims.d[1];

    // Greedy sampling
    if (mTopKSampling)
    {
        // 0. The first topK sampling workspace
        auto const draftTokenSamplingWorkspaceSize
            = getTopKWorkspaceSize<T>(numInputLogits, /* maxTokensPerStep */ 1, /* maxTopK */ maxTopK, vocabSizePadded);

        // 1. The first TopKs [numInputLogits]
        auto const topKsSize = numInputLogits * sizeof(SizeType32);

        // 2. Topks offset [batchSize]
        // Each request will have different number of logits that need to be sampled
        // This tensor will record the start offset of the topK for each request
        auto const topKOffsetSize = batchSize * sizeof(SizeType32);

        // 3. Logits ptrs [numInputLogits]
        auto const logitsPtrsSize = numInputLogits * sizeof(T*);

        // 4. The first topK sampling's output ids ptrs [numInputLogits][maxDecodingDraftTokens]
        auto const firstTopKOutputIdsPtrsSize = numInputLogits * sizeof(TokenIdType*);

        // 5. The first topK sampling's output ids (temporary buffer) [numInputLogits * maxDecodingDraftTokens]
        auto const firstTopKOutputIdsSize = numInputLogits * maxDecodingDraftTokens * sizeof(TokenIdType);

        // 6. Number of successors for each nodes, extract from the paths and layerId
        // [batchSize * maxDecodingTokens]
        auto const numSuccessorsForEachNodeSize = batchSize * maxDecodingTokens * sizeof(SizeType32);

        // 7. Flag whether to do decoding or not. SamplingTopK is done for numInputLogits tokens.
        // But only sum(numValidLogitsPerRequest[:]) of them are valid.
        // [batchSize * maxDecodingTokens]
        auto const skipDecodeSize = numInputLogits * sizeof(bool);

        // 8. The first topK sampling's logprobs [batchSize * maxDecodingDraftTokens]
        auto const firstTopKOutputLogProbsSize = numInputLogits * maxDecodingDraftTokens * sizeof(float);

        // 9. Eagle-2, the second topK sampling workspace
        // Sampling from [batchSize, maxTopK * maxTopK] to [batchSize, maxTopK]
        auto const secondTopKSamplingWorkspaceSize = getTopKWorkspaceSize<float>(
            batchSize, /* maxTokensPerStep */ 1, /* maxTopK */ maxTopK, maxTopK * maxTopK);

        // 10. Eagle-2, the outputIds of the second topK sampling, shape [batchSize, maxDecodingTokens]
        auto const secondTopKOutputIdsSize = batchSize * maxDecodingTokens * sizeof(TokenIdType);
        // 11. Eagle-2, the outputIdsPtr of the second topK sampling, shape [batchSize]
        auto const secondTopKOutputIdsPtrSize = batchSize * sizeof(TokenIdType*);
        // 12. Eagle-2, the inputScoresPtrs of the second topK sampling, shape [batchSize]
        auto const secondTopKInputScoresPtrsSize = batchSize * sizeof(float*);
        // 13. Eagle-2, the outpuLogProbs of the second topK samplig, shape [batchSize, maxDecodingDraftTokens]
        auto const secondTopKOutputLogProbsSize = batchSize * maxDecodingDraftTokens * sizeof(float);

        // 14. Eagle-2, the input scores pointers of the third topK sampling, shape [batchSize]
        // Each points to a vocabSize = '(mNumEagleLayers - 1) * dynamicTreeMaxTopK * dynamicTreeMaxTopK +
        // dynamicTreeMaxTopK'
        auto const thirdTopKInputScoresPtrsSize = batchSize * sizeof(float*);
        // 15. Eagle-2, the output of the third topK sampling, shape [batchSize, maxDecodingDraftTokens]
        auto const thirdTopKOutputIdsSize = batchSize * maxDecodingDraftTokens * sizeof(TokenIdType);
        // 16. Eagle-2, the output pointers of the third topK sampling, shape [batchSize]
        auto const thirdTopKOutputIdsPtrsSize = batchSize * sizeof(TokenIdType*);
        // 17. Eagle-2, the workspace of the third topK sampling
        // Sampling from [batchSize, '(mNumEagleLayers - 1) * dynamicTreeMaxTopK * dynamicTreeMaxTopK +
        // dynamicTreeMaxTopK'] to [batchSize, maxDecodingDraftTokens] We over-set the vocabsize here.
        auto const thridTopKSamplingWorkspaceSize = getTopKWorkspaceSize<float>(batchSize, /* maxTokensPerStep */ 1,
            /* maxTopK */ maxDecodingDraftTokens, mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens);

        // 18. Eagle-2, the topKs for each request in the third topK sampling
        // The real topK value is min(maxDecodingDraftTokens, totalNumDraftTokensForAllLayers)
        auto const thirdTopKsSize = batchSize * sizeof(SizeType32);

        SizeType32 constexpr NUM_BUFFERS{19};
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = draftTokenSamplingWorkspaceSize;
        workspaces[1] = topKsSize;
        workspaces[2] = topKOffsetSize;
        workspaces[3] = logitsPtrsSize;
        workspaces[4] = firstTopKOutputIdsPtrsSize;
        workspaces[5] = firstTopKOutputIdsSize;
        workspaces[6] = numSuccessorsForEachNodeSize;
        workspaces[7] = skipDecodeSize;
        workspaces[8] = firstTopKOutputLogProbsSize;
        workspaces[9] = secondTopKSamplingWorkspaceSize;
        workspaces[10] = secondTopKOutputIdsSize;
        workspaces[11] = secondTopKOutputIdsPtrSize;
        workspaces[12] = secondTopKInputScoresPtrsSize;
        workspaces[13] = secondTopKOutputLogProbsSize;
        workspaces[14] = thirdTopKInputScoresPtrsSize;
        workspaces[15] = thirdTopKOutputIdsSize;
        workspaces[16] = thirdTopKOutputIdsPtrsSize;
        workspaces[17] = thridTopKSamplingWorkspaceSize;
        workspaces[18] = thirdTopKsSize;
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

    // We allocate many buffers with 'numInputLogits' size, but the input logits will include some padding logits.
    // So only 'batchSize' or 'numValidLogits' size will be actually used.
    auto const numInputLogits = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const vocabSizePadded = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const batchSize = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[2];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    auto const maxTopK = maxDecodingDraftTokens;

    ////////////////////////////////////////// Get plugin inputs //////////////////////////////////////////
    // Plugin inputs
    // Input logits for sampling, shape: [numInputLogits, vocabSizePadded]
    auto pluginInputLogits = static_cast<T const*>(inputs[getIdx(InputIdxEntry::LOGITS)]);
    // Input paths, shape: [batchSize, maxDecodingTokens, maxPathLen]
    auto pluginInputPaths = static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PATHS)]);
    auto numValidLogits = static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::NUM_VALID_LOGITS)]);
    // For Eagle-2
    // Whether to use dynamic tree (i.e., Eagle-2)
    auto useDynamicTree = *(static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::USE_DYNAMIC_TREE)]));
    // The max topK for dynamic tree. All the requests have the same expand topK.
    // In Eagle-2, dynamicTreeMaxTopK is equal to maxNonLeavesPerLayer in the internal EagleNets.
    auto dynamicTreeMaxTopK = *(static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::DYNAMIC_TREE_MAX_TOPK)]));
    // All layer's draft tokenIds, shape: [batchSize, maxDecodingDraftTokens]
    auto pluginInputDraftTokenIds
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_DRAFT_TOKEN_IDS)]);
    // The number of all layer's draft tokenIds, shape: [batchSize]
    auto pluginInputDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::INPUT_DRAFT_LENS)]);
    // The previous EagleNet's scores, shape: [batchSize, maxDecodingDraftTokens]
    auto pluginInputPrevScores = static_cast<float const*>(inputs[getIdx(InputIdxEntry::INPUT_PREV_SCORES)]);
    // The indices of the nodes that will be expand in this layer, shape: [batchSize, maxDecodingDraftTokens]
    // The index is related to the final output tree, which has max_decoding_draft_tokens draft tokens.
    auto pluginInputCurrentExpandIndices
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_CURRENT_EXPAND_INDICES)]);
    // The scores from all previous EagleNets,
    // shape: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    auto pluginInputAllLayersScores = static_cast<float const*>(inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)]);
    // The draft tokens from all previous EagleNets,
    // shape: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    auto pluginInputAllLayersDraftTokenIds
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_DRAFT_TOKEN_IDS)]);
    // The predecessor of all the draft tokens,
    // shape: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    auto pluginInputAllLayersDraftTokenIdsPredecessor = reinterpret_cast<SizeType32 const*>(
        inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR)]);

    ////////////////////////////////////////// Get plugin outputs //////////////////////////////////////////
    // Plugin outputs
    // All layer's draft tokenIds, shape: [batchSize, maxDecodingDraftTokens]
    auto pluginOutputDraftTokenIds
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_DRAFT_TOKEN_IDS)]);
    // The number of all layer's draft tokenIds, shape: [batchSize]
    auto pluginOutputDraftLens = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_DRAFT_LENS)]);
    // For Eagle-2
    // Updated paths base on this layer's sampling result, shape: [batchSize, maxDecodingTokens, maxPathLen]
    auto pluginOutputPaths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_PATHS)]);
    // This layer's scores, which will be used in next layers [batchSize, maxDecodingDraftTokens]
    auto pluginOutputCurrentScores = static_cast<float*>(outputs[getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES)]);
    // The indices of the nodes that will be expand in next layer, shape: [batchSize, maxDecodingDraftTokens]
    // The index is related to the final output tree, which has max_decoding_draft_tokens draft tokens.
    auto pluginOutputNextExpandIndices
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_NEXT_EXPAND_INDICES)]);
    // Updated scores, shape: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    auto pluginOutputAllLayersScores = static_cast<float*>(outputs[getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES)]);
    // Updated draft tokens, shape: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x maxDecodingDraftTokens]
    auto pluginOutputAllLayersDraftTokenIds
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS)]);
    // Update the predecessor of the draft tokens, shape: [batchSize, mNumEagleLayers, maxDecodingDraftTokens x
    // maxDecodingDraftTokens]
    auto pluginOutputAllLayersDraftTokenIdsPredecessor
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR)]);

    ////////////////////////////////////////// Get workspaces //////////////////////////////////////////
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

    // Workspace 3: logits pointers tensor: shape: [numInputLogits]
    T const** logitsPtrs
        = reinterpret_cast<T const**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(T*)));

    // Workspace 4: outputIds pointers tensor: shape [numInputLogits], each points to a [maxDecodingDraftTokens] buffer
    TokenIdType** firstTopKOutputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(TokenIdType*)));

    // Workspace 5: outputIds tensor: flatten outputIds, shape [numInputLogits * maxDecodingDraftTokens]
    TokenIdType* firstTopKOutputIdsFlatten = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * maxDecodingDraftTokens * sizeof(TokenIdType)));

    // Workspace 6: number of successors for each nodes tensor: shape [batchSize * maxDecodingTokens]
    SizeType32* numSuccessorsForEachNode = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(SizeType32)));

    // Workspace 7: skip decoding mask [numInputLogits]
    bool* skipDecode
        = reinterpret_cast<bool*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(bool)));

    // In Eagle-1, we do not need to return logProbs
    float* firstTopKOutputLogProbs = nullptr;
    if (useDynamicTree)
    {
        // Workspace 8. The output logProbs of the first topK sampling.
        // Which will be updated with the previous layer's scores (i.e., pluginInputPrevScores), and will be treat as
        // the input of the second topK sampling. For mLayerIdx == 0, shape: [numInputLogits(batchSize),
        // maxDecodingDraftTokens] For mLayerIdx > 0, shape: [numInputLogits(batchSize * dynamicTreeMaxTopK),
        // maxDecodingDraftTokens]
        firstTopKOutputLogProbs = reinterpret_cast<float*>(
            tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * maxDecodingDraftTokens * sizeof(float)));
    }

    SizeType32 const secondTopKVocabSize = dynamicTreeMaxTopK * maxDecodingDraftTokens;
    // Workspace 9: Sampling from [batchSize, dynamicTreeMaxTopK * maxDecodingDraftTokens] to [batchSize,
    // dynamicTreeMaxTopK]
    auto const secondTopKSamplingWorkspaceSize
        = getTopKWorkspaceSize<float>(batchSize, /* maxTokensPerStep */ 1, /* maxTopK */ maxTopK, secondTopKVocabSize);
    void* workspaceScoresSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, secondTopKSamplingWorkspaceSize));

    // Workspace 10: the second (scores) sampling's outputIds, shape: [batchSize, maxDecodingDraftTokens]
    TokenIdType* secondTopKOutputIdsFlatten = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingDraftTokens * sizeof(TokenIdType)));

    // Workspace 11: the second (scores) sampling's outputIdsPtrs
    TokenIdType** secondTopKOutputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(TokenIdType*)));

    // Workspace 12: input scores pointers
    float** secondTopKInputScoresPtrs
        = reinterpret_cast<float**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(float*)));

    // Workspace 13: the second sampling's outputLogProbs
    float* secondTopKOutputLogProbs = reinterpret_cast<float*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingDraftTokens * sizeof(float)));

    // Workspace 14: The input scores pointers of the third topK sampling, shape [batchSize]
    float** thirdTopKInputScoresPtrs
        = reinterpret_cast<float**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(float*)));

    // Workspace 15: The output of the third topK sampling, shape [batchSize, maxDecodingDraftTokens]
    TokenIdType* thirdTopKOutputIds = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingDraftTokens * sizeof(TokenIdType)));

    // Workspace 16: The output pointers of the third topK sampling, shape [batchSize]
    TokenIdType** thirdTopKOutputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(TokenIdType*)));

    // The number of draft tokens among all layers
    long const totalNumDraftTokensForAllLayers
        = (mNumEagleLayers - 1) * dynamicTreeMaxTopK * dynamicTreeMaxTopK + dynamicTreeMaxTopK;

    auto const thridTopKSamplingWorkspaceSize = getTopKWorkspaceSize<float>(
        batchSize, /* maxTokensPerStep */ 1, /* maxTopK */ maxDecodingDraftTokens, totalNumDraftTokensForAllLayers);
    // Workspace 17: The workspace of the third topK sampling
    void* workspaceThirdTopKSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, thridTopKSamplingWorkspaceSize));

    // Workspace 18. Eagle-2, the topKs for each request in the third topK sampling, shape [batchSize]
    // The real topK value is min(maxDecodingDraftTokens, totalNumDraftTokensForAllLayers)
    SizeType32* thirdTopKs
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(SizeType32)));

    ////////////////////////////////////////// Main logic //////////////////////////////////////////
    // Fill logitsPtrs from plugin input logits
    // And fill firstTopKOutputIdsPtrs from firstTopKOutputIdsFlatten
    invokeAssembleDraftLogitsOffsets(logitsPtrs, pluginInputLogits, firstTopKOutputIdsPtrs, firstTopKOutputIdsFlatten,
        skipDecode, numValidLogits, numInputLogits, batchSize, maxDecodingDraftTokens, vocabSizePadded, stream);
    sync_check_cuda_error(stream);

    if (useDynamicTree)
    {
        // For Eagle-2, the topK value between different requests are the same, all set to 'dynamicTreeMaxTopK'.
        invokeSetTopKsFromDyanmicTreeMaxTopK(
            mLayerIdx, batchSize, numInputLogits, topKs, topKOffset, dynamicTreeMaxTopK, numValidLogits, stream);
        sync_check_cuda_error(stream);

        // Do softmax for the input logits
        // We set the 'batchSize' and 'maxBatchSize' to 'numInputLogits', while 'numInputLogits' logits may contain
        // some padding logits, which do not need to be calculated.
        // We use 'skipDecode' list to skip these padding logits. This could avoid redundant calculations.
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = const_cast<T*>(pluginInputLogits);
        biasSoftmaxParams.logitsPtrs = nullptr;
        biasSoftmaxParams.probs = const_cast<T*>(pluginInputLogits);
        biasSoftmaxParams.maxBeamWidth = 1;
        biasSoftmaxParams.batchSlots = nullptr;
        biasSoftmaxParams.batchSize = numInputLogits;
        biasSoftmaxParams.maxBatchSize = numInputLogits;
        biasSoftmaxParams.vocabSize = vocabSizePadded;
        biasSoftmaxParams.vocabSizePadded = vocabSizePadded;
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.skipDecode = skipDecode;
        biasSoftmaxParams.checkParams();

        invokeAddBiasSoftMax(biasSoftmaxParams, stream);
        sync_check_cuda_error(stream);
    }
    else
    {
        // For Eagle-1, extract topK value from input path.
        invokeExtractTopKsFromPath(pluginInputPaths, topKs, topKOffset, numSuccessorsForEachNode, mLayerIdx, batchSize,
            maxDecodingTokens, maxPathLen, stream);
        sync_check_cuda_error(stream);
    }

    TopKSamplingKernelParams<T> params{};
    params.logProbsPtrs = logitsPtrs;              // [numInputLogits][vocabSizePadded]
    params.outputIdsPtrs = firstTopKOutputIdsPtrs; // [numInputLogits][maxDecodingDraftTokens]
    params.workspace = workspaceSampling;
    params.maxTopK = maxTopK;
    params.topKs = topKs; // [numInputLogits]
    params.batchSize = numInputLogits;
    params.maxBatchSize = numInputLogits;
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = vocabSizePadded;
    params.returnAllSelectedTokens = true;
    params.strictTopPBoundary = false;
    params.skipDecode = skipDecode;
    params.outputLogProbs = firstTopKOutputLogProbs; // [numInputLogits * maxDecodingDraftTokens]
    params.logitsHasProbs = true;

    invokeBatchTopKSampling(params, stream);
    sync_check_cuda_error(stream);

    if (useDynamicTree)
    {
        // When mLayerIdx == 0, we do not need to update scores.
        // We take the outputLogProbs of the first topK sampling as the scores directly.
        if (mLayerIdx != 0)
        {
            // Update firstTopKOutputLogProbs with pluginInputPrevScores, which is the scores from the previous layer
            invokeUpdateScores(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens, firstTopKOutputLogProbs,
                pluginInputPrevScores, stream);
            sync_check_cuda_error(stream);

            // Do the second top-dynamicTreeMaxTopK sampling among this dynamicTreeMaxTopK x dynamicTreeMaxTopK draft
            // tokens. Through the second topK sampling, we obtain the dynamicTreeMaxTopK output draft tokens of this
            // layer.

            // Although theoretically we only need to select 'dynamicTreeMaxTopK' draft tokens from 'dynamicTreeMaxTopK
            // * dynamicTreeMaxTopK' draft tokens, we over-set vocabSize here. This is because when we write the scores
            // into firstTopKOutputLogProbs, we store it in the form of [batchSize * dynamicTreeMaxTopK,
            // maxDecodingDraftTokens]. For each request, these 'dynamicTreeMaxTopK * dynamicTreeMaxTopK' scores are not
            // saved continuously, but in the format of [dynamicTreeMaxTopK, maxDecodingDraftTokens]. For unused
            // positions, we set '-inf' to ensure that they will not be sampled. Examples: For a request,
            // dynamicTreeMaxTopK == 3, the scores in its buffer ([dynamicTreeMaxTopK, maxDecodingDraftTokens]) are as
            // follow:
            // [[1.1, 2.2, 3.3, -inf, -inf, ...],
            //  [4.4, 5.5, 6.6, -inf, -inf, ...],
            //  [7.7, 8.8, 9.9, -inf, -inf, ...]]

            // Prepare the input of the second topK sampling.
            invokeAssembleSecondTopKSamplingInputs(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
                firstTopKOutputLogProbs, secondTopKInputScoresPtrs, secondTopKOutputIdsFlatten, secondTopKOutputIdsPtrs,
                stream);
            sync_check_cuda_error(stream);

            TopKSamplingKernelParams<float> params{};
            params.logProbsPtrs = secondTopKInputScoresPtrs;
            params.outputIdsPtrs = secondTopKOutputIdsPtrs;
            params.workspace = workspaceScoresSampling;
            params.maxTopK = maxTopK; // Same to maxDecodingTokens
            params.topKs = topKs;     // [batchSize], all set to dynamicTreeMaxTopK
            params.batchSize = batchSize;
            params.maxBatchSize = batchSize;
            params.maxTokensPerStep = 1;
            params.vocabSizePadded = secondTopKVocabSize;
            params.returnAllSelectedTokens = true;
            params.strictTopPBoundary = false;

            invokeBatchTopKSampling(params, stream);
            sync_check_cuda_error(stream);
        }

        // Copy this layer's scores and draft tokensId:
        // 1) Copy this layer's scores to pluginOutputAllLayersScores
        // 2) Copy dynamicTreeMaxTopK (or dynamicTreeMaxTopK * dynamicTreeMaxTopK) draft tokens to
        // pluginOutputAllLayersDraftTokenIds 3) Set the predecessors of these draft tokens and save to
        // pluginOutputAllLayersDraftTokenIdsPredecessor,
        //    which will be used to reconstruct the final output tree at the last layer
        invokeCopyScoresAndDraftTokenIds(mLayerIdx, mNumEagleLayers, maxDecodingDraftTokens, batchSize,
            dynamicTreeMaxTopK,
            pluginInputCurrentExpandIndices, // The indices of the nodes that expand in this layer (i.e., the input
                                             // logits). The index is related to the final tree.
            pluginInputAllLayersScores, pluginInputAllLayersDraftTokenIds, pluginInputAllLayersDraftTokenIdsPredecessor,
            pluginOutputAllLayersScores, pluginOutputAllLayersDraftTokenIds,
            pluginOutputAllLayersDraftTokenIdsPredecessor,
            firstTopKOutputLogProbs,   // This layer's scores
            firstTopKOutputIdsFlatten, // This layer's draft tokens
            stream);
        sync_check_cuda_error(stream);

        // Update Path
        // For mLayerIdx == 0, the output of the first topK sampling are the output draft tokens of this layers. The
        // update logic is simple. For mLayerIdx > 0, the output of the second topK sampling are the output draft tokens
        // of this layers. 'secondTopKOutputIdsPtrs' contains the top-dynamicTreeMaxTopK selected from the second topK
        // sampling. 'pluginOutputNextExpandIndices' record the selected the top-dynamicTreeMaxTopK draft token's Id of
        // this layer,
        //     which will be used in the next layer to compute the predecessors.
        // The last layer will completely reconstruct the paths, so there is no need to update the paths here.
        if (mLayerIdx != mNumEagleLayers - 1)
        {
            invokeUpdatePath(mLayerIdx, batchSize, dynamicTreeMaxTopK, maxDecodingTokens, maxPathLen, pluginInputPaths,
                pluginOutputPaths,
                secondTopKOutputIdsPtrs, // if mLayerIdx == 0, secondTopKOutputIdsPtrs == nullptr, and it's useless
                                         // during update paths
                pluginOutputNextExpandIndices, stream);
            sync_check_cuda_error(stream);
        }

        if (mLayerIdx != 0)
        {
            // We will extract the real draft tokenIds and scores from 'firstTopKOutputIdsFlatten' and
            // 'secondTopKInputScoresPtrs' according to the 'secondTopKOutputIdsPtrs'. And store them into
            // 'secondTopKOutputIdsPtrs' and 'secondTopKOutputLogProbs' (reuse these buffers).
            // secondTopKInputScoresPtrs: shape [batchSize * dynamicTreeMaxTopK, maxDecodingDraftTokens]
            //     The original scores, which were used to do the second TopK sampling
            // secondTopKOutputIdsPtrs: shape [batchSize], each points to a [maxDecodingDraftTokens] buffer
            //     The output of the second TopK sampling, which are the indices of the top-dynamicTreeMaxTopK among
            //     'dynamicTreeMaxTopK * dynamicTreeMaxTopK'. We need to figure out what these top-dynamicTreeMaxTopK
            //     draft tokens' real tokenIds.
            // firstTopKOutputIdsFlatten: shape [batchSize * dynamicTreeMaxTopK, maxDecodingDraftTokens]
            //     The value are related to the vocabSize, which is the real tokenIds.
            invokeExtractScoresAndRealDraftTokensIds(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
                secondTopKInputScoresPtrs, secondTopKOutputIdsPtrs, firstTopKOutputIdsFlatten, secondTopKOutputLogProbs,
                stream);
            sync_check_cuda_error(stream);
        }

        // Copy this layer's output draft tokens and scores.
        // This layer's output scores is next layer's previous scores.
        // if mLayerIdx == 0, directly use the first topK's outputIds / logProbs as this layer's output draft tokens /
        // scores if mLayerIdx > 0, we use the second topK's outputIds / logProbs,
        //      which is updated with the real draft tokenIds / logprobs in 'invokeExtractScoresAndRealDraftTokensIds'
        invokeUpdateDraftTokensAndLensAndCurScores(mLayerIdx, batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
            mLayerIdx == 0 ? firstTopKOutputIdsPtrs : secondTopKOutputIdsPtrs, pluginInputDraftTokenIds,
            pluginInputDraftLens, pluginOutputDraftTokenIds, pluginOutputDraftLens,
            mLayerIdx == 0 ? firstTopKOutputLogProbs : secondTopKOutputLogProbs, pluginOutputCurrentScores, stream);
        sync_check_cuda_error(stream);

        if (mLayerIdx == mNumEagleLayers - 1)
        {
            // The maximum number of nodes on the final tree (exclude the root node)
            auto const maxNodesOnFinalTree = std::min(maxDecodingDraftTokens, totalNumDraftTokensForAllLayers);

            // When reach the last EagleNet, we need to do the third sampling, which take all layers' draft tokens and
            // scores as input, and then select top-maxDecodingDraftTokens draft tokens among them. We need to
            // reconstruct the path/tree after the third topK sampling.
            invokeAssembleThridTopKSamplingInputs(batchSize, maxDecodingDraftTokens, mNumEagleLayers,
                maxNodesOnFinalTree, thirdTopKs, pluginOutputAllLayersScores, thirdTopKInputScoresPtrs,
                thirdTopKOutputIds, thirdTopKOutputIdsPtrs, stream);
            sync_check_cuda_error(stream);

            // 1) Do topK sampling among all previous draft tokens
            TopKSamplingKernelParams<float> params{};
            params.logProbsPtrs = thirdTopKInputScoresPtrs;
            params.outputIdsPtrs = thirdTopKOutputIdsPtrs;
            params.workspace = workspaceThirdTopKSampling;
            params.topKs = thirdTopKs;               // All set to 'maxNodesOnFinalTree'
            params.maxTopK = maxDecodingDraftTokens; // We set maxTopK to 'maxDecodingDraftTokens' to align the
                                                     // outputIdsPtrs offsets when written back.
            params.batchSize = batchSize;
            params.maxBatchSize = batchSize;
            params.maxTokensPerStep = 1;
            params.vocabSizePadded = totalNumDraftTokensForAllLayers;
            params.returnAllSelectedTokens = true;
            params.strictTopPBoundary = false; // Make sure to select topK tokens.

            invokeBatchTopKSampling(params, stream);
            sync_check_cuda_error(stream);

            // 2) Reconstruct the Path
            invokeReconstructFinalPath(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens, maxDecodingTokens,
                maxPathLen, mNumEagleLayers, maxNodesOnFinalTree, thirdTopKOutputIdsPtrs,
                pluginOutputAllLayersDraftTokenIdsPredecessor, pluginOutputPaths, stream);
            sync_check_cuda_error(stream);

            // 3) Copy this layer's outputIds to outputDraftTokenIds
            invokeCopyFinalDraftTokens(batchSize, maxDecodingDraftTokens, mNumEagleLayers, maxNodesOnFinalTree,
                thirdTopKOutputIdsPtrs, pluginOutputAllLayersDraftTokenIds, pluginOutputDraftTokenIds,
                pluginOutputDraftLens, stream);
            sync_check_cuda_error(stream);
        }
    }
    else
    {
        // Eagle-1: Copy output token id from outputIdsPtrs to the plugin output buffer
        invokeCopyOutputTokensIds(firstTopKOutputIdsPtrs, topKs, topKOffset, pluginInputDraftTokenIds,
            pluginInputDraftLens, numValidLogits, pluginOutputDraftTokenIds, pluginOutputDraftLens, mLayerIdx,
            batchSize, maxDecodingDraftTokens, pluginInputPaths, pluginOutputPaths, maxPathLen, stream);
        sync_check_cuda_error(stream);
    }

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
    TLLM_CHECK(index < getNbOutputs());
    TLLM_CHECK(index < getNbOutputs());
    if (index == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES)
        || index == getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES))
    {
        // Only output_prev_socres are float
        return inputTypes[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)];
    }
    else
    {
        // output_draft_token_ids, output_draft_lens, output_paths, output_next_expand_index,
        // output_all_layers_draft_token_ids, output_all_layers_draft_token_ids_predecessor
        // are all int32 type, same as path
        return inputTypes[getIdx(InputIdxEntry::PATHS)];
    }
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
    return 8;
}

int EagleDecodeDraftTokensPlugin::initialize() noexcept
{
    return 0;
}

void EagleDecodeDraftTokensPlugin::terminate() noexcept {}

size_t EagleDecodeDraftTokensPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDtype) + sizeof(mLayerIdx) + sizeof(mNumEagleLayers) + sizeof(mTopKSampling);
}

void EagleDecodeDraftTokensPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDtype);
    write(d, mLayerIdx);
    write(d, mNumEagleLayers);
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
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_eagle_layers", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("top_k_sampling", nullptr, PluginFieldType::kINT32));
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
    int32_t layerIdx{};
    int32_t numEagleLayers{};
    nvinfer1::DataType type{};
    bool topKSampling{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "layer_idx"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            layerIdx = *static_cast<int32_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "num_eagle_layers"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            numEagleLayers = *static_cast<int32_t const*>(fields[i].data);
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
        auto* obj = new EagleDecodeDraftTokensPlugin(type, layerIdx, numEagleLayers, topKSampling);
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
