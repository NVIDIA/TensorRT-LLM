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
#include "tensorrt_llm/runtime/iTensor.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::IdentityPluginCreator;
using tensorrt_llm::plugins::IdentityPlugin;

static char const* IDENTITY_PLUGIN_VERSION{"1"};
static char const* IDENTITY_PLUGIN_NAME{"Identity"};
PluginFieldCollection IdentityPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> IdentityPluginCreator::mPluginAttributes;

IdentityPlugin::IdentityPlugin() {}

// Parameterized constructor
IdentityPlugin::IdentityPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
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
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool IdentityPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(0 <= pos && pos < 2);
    PluginTensorDesc const& input = inOut[0];
    PluginTensorDesc const& output = inOut[1];
    switch (pos)
    {
    case 0: return input.format == nvinfer1::TensorFormat::kLINEAR;
    case 1: return output.type == input.type && output.format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

void IdentityPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t IdentityPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int IdentityPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    size_t count = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        count *= inputDesc[0].dims.d[i];
    }
    count *= tensorrt_llm::runtime::BufferDataType(inputDesc[0].type).getSize();

    cudaMemcpyAsync(outputs[0], inputs[0], count, cudaMemcpyDeviceToDevice, stream);

    sync_check_cuda_error(stream);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType IdentityPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* IdentityPlugin::getPluginType() const noexcept
{
    return IDENTITY_PLUGIN_NAME;
}

char const* IdentityPlugin::getPluginVersion() const noexcept
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

char const* IdentityPluginCreator::getPluginName() const noexcept
{
    return IDENTITY_PLUGIN_NAME;
}

char const* IdentityPluginCreator::getPluginVersion() const noexcept
{
    return IDENTITY_PLUGIN_VERSION;
}

PluginFieldCollection const* IdentityPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* IdentityPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* obj = new IdentityPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* IdentityPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call IdentityPlugin::destroy()
    try
    {
        auto* obj = new IdentityPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
