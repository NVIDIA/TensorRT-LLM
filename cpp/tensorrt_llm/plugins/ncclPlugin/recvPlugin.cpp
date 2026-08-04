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
#include "recvPlugin.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <nccl.h>

using namespace nvinfer1;
using tensorrt_llm::plugins::RecvPluginCreator;
using tensorrt_llm::plugins::RecvPlugin;
using tensorrt_llm::mpi::MpiTag;

static char const* RECV_PLUGIN_VERSION{"1"};
static char const* RECV_PLUGIN_NAME{"Recv"};
PluginFieldCollection RecvPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RecvPluginCreator::mPluginAttributes;

RecvPlugin::RecvPlugin(int srcRank, nvinfer1::DataType type)
    : mSrcRank(srcRank)
    , mType(type)
{
}

// Parameterized constructor
RecvPlugin::RecvPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mSrcRank);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RecvPlugin::clone() const noexcept
{
    auto* plugin = new RecvPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RecvPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool RecvPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RecvPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t RecvPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RecvPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
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
    TLLM_LOG_DEBUG("start ncclRecv with size %d", size);
    NCCLCHECK(ncclRecv(outputs[0], size, (*getDtypeMap())[inputDesc[0].type], 0, mComm, stream));
    TLLM_LOG_DEBUG("end ncclRecv with size %d", size);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RecvPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* RecvPlugin::getPluginType() const noexcept
{
    return RECV_PLUGIN_NAME;
}

char const* RecvPlugin::getPluginVersion() const noexcept
{
    return RECV_PLUGIN_VERSION;
}

int RecvPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int RecvPlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    ncclUniqueId id;
    COMM_SESSION.recvValue(id, mSrcRank, MpiTag::kDefault);
// Need static connection initialization for accurate KV cache size estimation
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
#endif // _WIN32
    NCCLCHECK(ncclCommInitRank(&mComm, 2, id, 1));
    return 0;
}

void RecvPlugin::terminate() noexcept
{
    if (isBuilding())
    {
        return;
    }
    NCCLCHECK(ncclCommDestroy(mComm));
}

size_t RecvPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSrcRank) + sizeof(mType);
}

void RecvPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mSrcRank);
    TLLM_CHECK(d == a + getSerializationSize());
}

void RecvPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

RecvPluginCreator::RecvPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("src_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* RecvPluginCreator::getPluginName() const noexcept
{
    return RECV_PLUGIN_NAME;
}

char const* RecvPluginCreator::getPluginVersion() const noexcept
{
    return RECV_PLUGIN_VERSION;
}

PluginFieldCollection const* RecvPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RecvPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int srcRank{};
    nvinfer1::DataType type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "src_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            srcRank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new RecvPlugin(srcRank, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RecvPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RecvPlugin::destroy()
    try
    {
        auto* obj = new RecvPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
