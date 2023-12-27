/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "sendPlugin.h"

#include "tensorrt_llm/common/mpiUtils.h"

#include <cassert>
#include <nccl.h>

using namespace nvinfer1;
using tensorrt_llm::plugins::SendPluginCreator;
using tensorrt_llm::plugins::SendPlugin;

static const char* SEND_PLUGIN_VERSION{"1"};
static const char* SEND_PLUGIN_NAME{"Send"};
PluginFieldCollection SendPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SendPluginCreator::mPluginAttributes;

SendPlugin::SendPlugin(int tgtRank, nvinfer1::DataType type)
    : mTgtRank(tgtRank)
    , mType(type)
{
}

// Parameterized constructor
SendPlugin::SendPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, mTgtRank);
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SendPlugin::clone() const noexcept
{
    auto* plugin = new SendPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs SendPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool SendPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void SendPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t SendPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int SendPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
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

    NCCLCHECK(ncclSend(inputs[0], size, (*getDtypeMap())[inputDesc[0].type], 1, mComm, stream));
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType SendPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* SendPlugin::getPluginType() const noexcept
{
    return SEND_PLUGIN_NAME;
}

const char* SendPlugin::getPluginVersion() const noexcept
{
    return SEND_PLUGIN_VERSION;
}

int SendPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int SendPlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    COMM_SESSION.send(id, mTgtRank, 0);
    NCCLCHECK(ncclCommInitRank(&mComm, 2, id, 0));
    return 0;
}

void SendPlugin::terminate() noexcept
{
    if (isBuilding())
    {
        return;
    }
    NCCLCHECK(ncclCommDestroy(mComm));
}

size_t SendPlugin::getSerializationSize() const noexcept
{
    return sizeof(mTgtRank) + sizeof(mType);
}

void SendPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mTgtRank);
    assert(d == a + getSerializationSize());
}

void SendPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

SendPluginCreator::SendPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("tgt_rank", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SendPluginCreator::getPluginName() const noexcept
{
    return SEND_PLUGIN_NAME;
}

const char* SendPluginCreator::getPluginVersion() const noexcept
{
    return SEND_PLUGIN_VERSION;
}

const PluginFieldCollection* SendPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SendPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int tgtRank;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "tgt_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            tgtRank = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new SendPlugin(tgtRank, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SendPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SendPlugin::destroy()
    try
    {
        auto* obj = new SendPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
