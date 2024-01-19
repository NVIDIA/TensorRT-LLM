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
#include "allgatherPlugin.h"

#include <nccl.h>

using namespace nvinfer1;
using tensorrt_llm::plugins::AllgatherPluginCreator;
using tensorrt_llm::plugins::AllgatherPlugin;

static const char* ALLGATHER_PLUGIN_VERSION{"1"};
static const char* ALLGATHER_PLUGIN_NAME{"AllGather"};
PluginFieldCollection AllgatherPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AllgatherPluginCreator::mPluginAttributes;

AllgatherPlugin::AllgatherPlugin(std::set<int> group, nvinfer1::DataType type)
    : mGroup(std::move(group))
    , mType(type)
{
}

// Parameterized constructor
AllgatherPlugin::AllgatherPlugin(const void* data, size_t length)
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
nvinfer1::IPluginV2DynamicExt* AllgatherPlugin::clone() const noexcept
{
    auto* plugin = new AllgatherPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AllgatherPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    auto ret = inputs[0];
    auto groupSize = exprBuilder.constant(mGroup.size());
    ret.d[0] = exprBuilder.operation(DimensionOperation::kPROD, *ret.d[0], *groupSize);
    return ret;
}

bool AllgatherPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{

    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void AllgatherPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t AllgatherPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int AllgatherPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }

    NCCLCHECK(ncclAllGather(
        inputs[0], outputs[0], size, (*getDtypeMap())[inputDesc[0].type], (*getCommMap())[mGroup], stream));

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType AllgatherPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* AllgatherPlugin::getPluginType() const noexcept
{
    return ALLGATHER_PLUGIN_NAME;
}

const char* AllgatherPlugin::getPluginVersion() const noexcept
{
    return ALLGATHER_PLUGIN_VERSION;
}

int AllgatherPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int AllgatherPlugin::initialize() noexcept
{
    initCommMap(mGroup);
    return 0;
}

void AllgatherPlugin::terminate() noexcept
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

size_t AllgatherPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType);
}

void AllgatherPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void AllgatherPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

AllgatherPluginCreator::AllgatherPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* AllgatherPluginCreator::getPluginName() const noexcept
{
    return ALLGATHER_PLUGIN_NAME;
}

const char* AllgatherPluginCreator::getPluginVersion() const noexcept
{
    return ALLGATHER_PLUGIN_VERSION;
}

const PluginFieldCollection* AllgatherPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AllgatherPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
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
        auto* obj = new AllgatherPlugin(group, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AllgatherPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call AllgatherPlugin::destroy()
    try
    {
        auto* obj = new AllgatherPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
