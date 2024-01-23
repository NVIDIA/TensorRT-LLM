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
#include "identityPlugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::IdentityPluginCreator;
using tensorrt_llm::plugins::IdentityPlugin;

static const char* IDENTITY_PLUGIN_VERSION{"1"};
static const char* IDENTITY_PLUGIN_NAME{"Identity"};
PluginFieldCollection IdentityPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> IdentityPluginCreator::mPluginAttributes;

IdentityPlugin::IdentityPlugin() {}

// Parameterized constructor
IdentityPlugin::IdentityPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* IdentityPlugin::clone() const noexcept
{
    auto* plugin = new IdentityPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs IdentityPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool IdentityPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(0 <= pos && pos < 2);
    const PluginTensorDesc& input = inOut[0];
    const PluginTensorDesc& output = inOut[1];
    switch (pos)
    {
    case 0: return input.format == nvinfer1::TensorFormat::kLINEAR;
    case 1: return output.type == input.type && output.format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

void IdentityPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t IdentityPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int IdentityPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    size_t count = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        count *= inputDesc[0].dims.d[i];
    }
    count *= tensorrt_llm::runtime::BufferDataType(inputDesc[0].type).getSize();

    cudaMemcpyAsync(outputs[0], inputs[0], count, cudaMemcpyDeviceToDevice, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType IdentityPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* IdentityPlugin::getPluginType() const noexcept
{
    return IDENTITY_PLUGIN_NAME;
}

const char* IdentityPlugin::getPluginVersion() const noexcept
{
    return IDENTITY_PLUGIN_VERSION;
}

int IdentityPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int IdentityPlugin::initialize() noexcept
{
    return 0;
}

void IdentityPlugin::terminate() noexcept {}

size_t IdentityPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void IdentityPlugin::serialize(void* buffer) const noexcept {}

void IdentityPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

IdentityPluginCreator::IdentityPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* IdentityPluginCreator::getPluginName() const noexcept
{
    return IDENTITY_PLUGIN_NAME;
}

const char* IdentityPluginCreator::getPluginVersion() const noexcept
{
    return IDENTITY_PLUGIN_VERSION;
}

const PluginFieldCollection* IdentityPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* IdentityPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* obj = new IdentityPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* IdentityPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call IdentityPlugin::destroy()
    try
    {
        auto* obj = new IdentityPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
