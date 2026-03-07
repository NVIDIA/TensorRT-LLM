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
#include "aetherSparseAttentionPlugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::AetherSparseAttentionPluginCreator;
using tensorrt_llm::plugins::AetherSparseAttentionPlugin;

static char const* AETHER_PLUGIN_VERSION{"1"};
static char const* AETHER_PLUGIN_NAME{"AetherSparseAttention"};
PluginFieldCollection AetherSparseAttentionPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AetherSparseAttentionPluginCreator::mPluginAttributes;

AetherSparseAttentionPlugin::AetherSparseAttentionPlugin() {}

AetherSparseAttentionPlugin::AetherSparseAttentionPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d).",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* AetherSparseAttentionPlugin::clone() const noexcept
{
    auto* plugin = new AetherSparseAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AetherSparseAttentionPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output is same dimensions as Query (inputs[0])
    return inputs[0];
}

bool AetherSparseAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void AetherSparseAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t AetherSparseAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // Workspace needed for index compaction
    return 0; // TODO: Calculate required workspace size
}

int AetherSparseAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // TODO: Dispatch custom CUDA kernels:
    // 1. aether_generate_indices
    // 2. aether_sparse_flash_fwd_kernel
    
    // For scaffolding, just return success
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType AetherSparseAttentionPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods
char const* AetherSparseAttentionPlugin::getPluginType() const noexcept
{
    return AETHER_PLUGIN_NAME;
}

char const* AetherSparseAttentionPlugin::getPluginVersion() const noexcept
{
    return AETHER_PLUGIN_VERSION;
}

int AetherSparseAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int AetherSparseAttentionPlugin::initialize() noexcept
{
    return 0;
}

void AetherSparseAttentionPlugin::terminate() noexcept {}

size_t AetherSparseAttentionPlugin::getSerializationSize() const noexcept
{
    return 0; // TODO: Serialize configuration parameters if any
}

void AetherSparseAttentionPlugin::serialize(void* buffer) const noexcept {}

void AetherSparseAttentionPlugin::destroy() noexcept
{
    delete this;
}

// Creator Methods
AetherSparseAttentionPluginCreator::AetherSparseAttentionPluginCreator()
{
    mPluginAttributes.clear();
    // Add specific fields if required
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AetherSparseAttentionPluginCreator::getPluginName() const noexcept
{
    return AETHER_PLUGIN_NAME;
}

char const* AetherSparseAttentionPluginCreator::getPluginVersion() const noexcept
{
    return AETHER_PLUGIN_VERSION;
}

PluginFieldCollection const* AetherSparseAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AetherSparseAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* obj = new AetherSparseAttentionPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AetherSparseAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new AetherSparseAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
