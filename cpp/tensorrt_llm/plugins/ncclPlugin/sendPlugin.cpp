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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <cassert>
#include <nccl.h>

using namespace nvinfer1;
using tensorrt_llm::plugins::SendPluginCreator;
using tensorrt_llm::plugins::SendPlugin;
using tensorrt_llm::mpi::MpiTag;

static char const* SEND_PLUGIN_VERSION{"1"};
static char const* SEND_PLUGIN_NAME{"Send"};
PluginFieldCollection SendPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SendPluginCreator::mPluginAttributes;

SendPlugin::SendPlugin(int tgtRank, nvinfer1::DataType type)
    : mTgtRank(tgtRank)
    , mType(type)
{
}

// Parameterized constructor
SendPlugin::SendPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mTgtRank);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SendPlugin::clone() const noexcept
{
    auto* plugin = new SendPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs SendPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool SendPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void SendPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t SendPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int SendPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }

    TLLM_LOG_DEBUG("start ncclSend with size %d", size);
    NCCLCHECK(ncclSend(inputs[0], size, (*getDtypeMap())[inputDesc[0].type], 1, mComm, stream));
    TLLM_LOG_DEBUG("end ncclSend with size %d", size);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType SendPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* SendPlugin::getPluginType() const noexcept
{
    return SEND_PLUGIN_NAME;
}

char const* SendPlugin::getPluginVersion() const noexcept
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
    COMM_SESSION.sendValue(id, mTgtRank, MpiTag::kDefault);
// Need static connection initialization for accurate KV cache size estimation
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
#endif // _WIN32
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
    TLLM_CHECK(d == a + getSerializationSize());
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
    mPluginAttributes.emplace_back(PluginField("tgt_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SendPluginCreator::getPluginName() const noexcept
{
    return SEND_PLUGIN_NAME;
}

char const* SendPluginCreator::getPluginVersion() const noexcept
{
    return SEND_PLUGIN_VERSION;
}

PluginFieldCollection const* SendPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SendPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int tgtRank{};
    nvinfer1::DataType type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "tgt_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            tgtRank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new SendPlugin(tgtRank, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SendPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SendPlugin::destroy()
    try
    {
        auto* obj = new SendPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
