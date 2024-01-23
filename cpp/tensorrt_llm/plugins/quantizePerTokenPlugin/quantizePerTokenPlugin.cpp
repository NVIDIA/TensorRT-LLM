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
#include "quantizePerTokenPlugin.h"
#include "tensorrt_llm/kernels/quantization.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::plugins::QuantizePerTokenPluginCreator;
using tensorrt_llm::plugins::QuantizePerTokenPlugin;

static const char* QUANTIZE_PER_TOKEN_PLUGIN_VERSION{"1"};
static const char* QUANTIZE_PER_TOKEN_PLUGIN_NAME{"QuantizePerToken"};
PluginFieldCollection QuantizePerTokenPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizePerTokenPluginCreator::mPluginAttributes;

QuantizePerTokenPlugin::QuantizePerTokenPlugin() {}

// Parameterized constructor
QuantizePerTokenPlugin::QuantizePerTokenPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QuantizePerTokenPlugin::clone() const noexcept
{
    auto* plugin = new QuantizePerTokenPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs QuantizePerTokenPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 1);
        TLLM_CHECK(outputIndex < 2);
        if (outputIndex == 0)
        {
            // Quantized input
            return inputs[0];
        }

        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int ii = 0; ii < ret.nbDims - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[ret.nbDims - 1] = exprBuilder.constant(1);
        // [M(*), 1] dynamic per token scales
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QuantizePerTokenPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
            && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // quantized activation
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // scales
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }
}

void QuantizePerTokenPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t QuantizePerTokenPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int QuantizePerTokenPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     activation     [M(*), K]
    // outputs
    //     quant          [M(*), K]
    //     scale_tokens   [M(*), 1]

    int64_t m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int64_t k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        invokePerTokenQuantization<float>(reinterpret_cast<int8_t*>(outputs[0]),
            reinterpret_cast<const float*>(inputs[0]), m, k, reinterpret_cast<float*>(outputs[1]), stream);
    }
    else
    {
        invokePerTokenQuantization<half>(reinterpret_cast<int8_t*>(outputs[0]),
            reinterpret_cast<const half*>(inputs[0]), m, k, reinterpret_cast<float*>(outputs[1]), stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType QuantizePerTokenPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(nbInputs == 1);
    TLLM_CHECK(index < 2);
    return index == 0 ? nvinfer1::DataType::kINT8 : nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods

const char* QuantizePerTokenPlugin::getPluginType() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_NAME;
}

const char* QuantizePerTokenPlugin::getPluginVersion() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_VERSION;
}

int QuantizePerTokenPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int QuantizePerTokenPlugin::initialize() noexcept
{
    return 0;
}

void QuantizePerTokenPlugin::terminate() noexcept {}

size_t QuantizePerTokenPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void QuantizePerTokenPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

void QuantizePerTokenPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

QuantizePerTokenPluginCreator::QuantizePerTokenPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QuantizePerTokenPluginCreator::getPluginName() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_NAME;
}

const char* QuantizePerTokenPluginCreator::getPluginVersion() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_VERSION;
}

const PluginFieldCollection* QuantizePerTokenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizePerTokenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        auto* obj = new QuantizePerTokenPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizePerTokenPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QuantizePerTokenPlugin::destroy()
    try
    {
        auto* obj = new QuantizePerTokenPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
