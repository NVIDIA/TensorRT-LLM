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

using namespace nvinfer1;
using tensorrt_llm::plugins::EagleDecodeDraftTokensPluginCreator;
using tensorrt_llm::plugins::EagleDecodeDraftTokensPlugin;

static char const* EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION{"1"};
static char const* EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME{"EagleDecodeDraftTokens"};
PluginFieldCollection EagleDecodeDraftTokensPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EagleDecodeDraftTokensPluginCreator::mPluginAttributes;

EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(nvinfer1::DataType type, int32_t layerIdx)
    : mDtype(type)
    , mLayerIdx(layerIdx)
{
}

// Parameterized constructor
EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDtype);
    read(d, mLayerIdx);
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
    TLLM_CHECK(nbInputs == 5);
    return inputs[outputIndex + 1];
}

bool EagleDecodeDraftTokensPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == 0) // logits
    {
        return (inOut[pos].type == mDtype) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == 3) // rand_data_sample
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else // next_draft_tokens, next_draft_lens, paths, tree_indices
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void EagleDecodeDraftTokensPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t EagleDecodeDraftTokensPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int EagleDecodeDraftTokensPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // TODO fill me

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType EagleDecodeDraftTokensPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index < 2);
    return inputTypes[index + 1];
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
    return sizeof(mDtype) + sizeof(mLayerIdx);
}

void EagleDecodeDraftTokensPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mLayerIdx);
    write(d, mDtype);
    assert(d == a + getSerializationSize());
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
    }

    try
    {
        auto* obj = new EagleDecodeDraftTokensPlugin(type, layerIdx);
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
