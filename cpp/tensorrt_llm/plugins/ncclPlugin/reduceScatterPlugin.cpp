/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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
#include "reduceScatterPlugin.h"

#include <cassert>
#include <nccl.h>

using namespace nvinfer1;
using tensorrt_llm::plugins::ReduceScatterPluginCreator;
using tensorrt_llm::plugins::ReduceScatterPlugin;

static const char* REDUCE_SCATTER_PLUGIN_VERSION{"1"};
static const char* REDUCE_SCATTER_PLUGIN_NAME{"ReduceScatter"};
PluginFieldCollection ReduceScatterPluginCreator::mFC{};
std::vector<PluginField> ReduceScatterPluginCreator::mPluginAttributes;

ReduceScatterPlugin::ReduceScatterPlugin(std::set<int> group, nvinfer1::DataType type)
    : mGroup(std::move(group))
    , mType(type)
{
}

// Parameterized constructor
ReduceScatterPlugin::ReduceScatterPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* ReduceScatterPlugin::clone() const noexcept
{
    auto* plugin = new ReduceScatterPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs ReduceScatterPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    auto output = inputs[0];
    output.d[0]
        = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *output.d[0], *exprBuilder.constant(mGroup.size()));
    return output;
}

bool ReduceScatterPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void ReduceScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t ReduceScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int ReduceScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < outputDesc[0].dims.nbDims; ++i)
    {
        size *= outputDesc[0].dims.d[i];
    }

    NCCLCHECK(ncclReduceScatter(
        inputs[0], outputs[0], size, (*getDtypeMap())[inputDesc[0].type], ncclSum, (*getCommMap())[mGroup], stream));

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType ReduceScatterPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* ReduceScatterPlugin::getPluginType() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_NAME;
}

const char* ReduceScatterPlugin::getPluginVersion() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_VERSION;
}

int ReduceScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int ReduceScatterPlugin::initialize() noexcept
{
    initCommMap(mGroup);
    return 0;
}

void ReduceScatterPlugin::terminate() noexcept
{
    auto* commMap = getCommMap();
    // [] operator inserts T() if it does not exist
    if (isBuilding() || (*commMap)[mGroup] == nullptr)
    {
        return;
    }
    NCCLCHECK(ncclCommDestroy((*commMap)[mGroup]));
    (*commMap)[mGroup] = nullptr;
}

size_t ReduceScatterPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType);
}

void ReduceScatterPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void ReduceScatterPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

ReduceScatterPluginCreator::ReduceScatterPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ReduceScatterPluginCreator::getPluginName() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_NAME;
}

const char* ReduceScatterPluginCreator::getPluginVersion() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_VERSION;
}

const PluginFieldCollection* ReduceScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ReduceScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            const auto* r = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new ReduceScatterPlugin(group, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ReduceScatterPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call ReduceScatterPlugin::destroy()
    try
    {
        auto* obj = new ReduceScatterPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
