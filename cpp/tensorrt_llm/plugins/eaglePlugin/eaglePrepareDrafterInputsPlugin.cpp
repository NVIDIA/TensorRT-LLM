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

using namespace nvinfer1;
using tensorrt_llm::plugins::EaglePrepareDrafterInputsPluginCreator;
using tensorrt_llm::plugins::EaglePrepareDrafterInputsPlugin;

static char const* EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_VERSION{"1"};
static char const* EAGLE_PREPARE_DRAFTER_INPUTS_PLUGIN_NAME{"EaglePrepareDrafterInputs"};
PluginFieldCollection EaglePrepareDrafterInputsPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EaglePrepareDrafterInputsPluginCreator::mPluginAttributes;

EaglePrepareDrafterInputsPlugin::EaglePrepareDrafterInputsPlugin(nvinfer1::DataType type, int32_t layerIdx)
    : mDtype(type)
    , mLayerIdx(layerIdx)
{
}

// Parameterized constructor
EaglePrepareDrafterInputsPlugin::EaglePrepareDrafterInputsPlugin(void const* data, size_t length)
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
nvinfer1::IPluginV2DynamicExt* EaglePrepareDrafterInputsPlugin::clone() const noexcept
{
    auto* plugin = new EaglePrepareDrafterInputsPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs EaglePrepareDrafterInputsPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex < 10);
    TLLM_CHECK(nbInputs == 7);
    auto const batchSizeExpr = inputs[nbInputs - 2].d[0];
    auto const maxDraftLenExpr = inputs[nbInputs - 2].d[1];

    nvinfer1::DimsExprs ret;
    switch (outputIndex)
    {
    case 0: // sequence_length
    case 1: // host_request_types
    case 2: // host_past_key_value_lengths
        ret = inputs[outputIndex];
        break;
    case 3: // spec_decoding_generation_lengths
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
        break;
    case 4: // spec_decoding_position_offsets
    case 5: // input_ids
    case 6: // position_ids
        // FIXME input_ids should have real value, not maxDraftLen
        ret.nbDims = 1;
        ret.d[0] = maxDraftLenExpr;
        break;
    case 7: // spec_decoding_packed_mask
        // FIXME
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDraftLenExpr;
        ret.d[2] = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *maxDraftLenExpr, *exprBuilder.constant(32));
        break;
    case 8: // hidden_dim
        ret.nbDims = 2;
        // FIXME real dim instead of max draft len
        ret.d[0] = maxDraftLenExpr;
        ret.d[1] = inputs[4].d[1];
        break;
    }
    return ret;
}

bool EaglePrepareDrafterInputsPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == nbInputs - 1 || pos == nbInputs + nbOutputs - 1) // hidden_states
    {
        return (inOut[pos].type == mDtype) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == 3) // kv cache pool pointers
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else // all other tensors
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void EaglePrepareDrafterInputsPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t EaglePrepareDrafterInputsPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int EaglePrepareDrafterInputsPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // TODO fill me

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType EaglePrepareDrafterInputsPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index < 9);
    if (index < 8)
    {
        return inputTypes[0];            // type of sequence_length
    }
    else                                 // hidden_states
    {
        return inputTypes[nbInputs - 1]; // type of hidden_states
    }
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
    return 9;
}

int EaglePrepareDrafterInputsPlugin::initialize() noexcept
{
    return 0;
}

void EaglePrepareDrafterInputsPlugin::terminate() noexcept {}

size_t EaglePrepareDrafterInputsPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDtype) + sizeof(mLayerIdx);
}

void EaglePrepareDrafterInputsPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mLayerIdx);
    write(d, mDtype);
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
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
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
        auto* obj = new EaglePrepareDrafterInputsPlugin(type, layerIdx);
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
