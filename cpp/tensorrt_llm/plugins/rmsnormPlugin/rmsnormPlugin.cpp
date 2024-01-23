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
#include "rmsnormPlugin/rmsnormPlugin.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::RmsnormPluginCreator;
using tensorrt_llm::plugins::RmsnormPlugin;

static const char* RMSNORM_PLUGIN_VERSION{"1"};
static const char* RMSNORM_PLUGIN_NAME{"Rmsnorm"};
PluginFieldCollection RmsnormPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RmsnormPluginCreator::mPluginAttributes;

RmsnormPlugin::RmsnormPlugin(float eps, nvinfer1::DataType type)
    : mEps(eps)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

// Parameterized constructor
RmsnormPlugin::RmsnormPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    read(d, mType);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RmsnormPlugin::clone() const noexcept
{
    auto* plugin = new RmsnormPlugin(mEps, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RmsnormPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool RmsnormPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < 5);
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RmsnormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RmsnormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RmsnormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     weight [N, ]
    // outputs
    //     output [M(*), N]

    int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];

    if (mType == DataType::kHALF)
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, (half*) nullptr, mEps, m, n, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* input = reinterpret_cast<const float*>(inputs[0]);
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, (float*) nullptr, mEps, m, n, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        const __nv_bfloat16* input = reinterpret_cast<const __nv_bfloat16*>(inputs[0]);
        const __nv_bfloat16* weight = reinterpret_cast<const __nv_bfloat16*>(inputs[1]);
        __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, (__nv_bfloat16*) nullptr, mEps, m, n, stream);
    }
#endif

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RmsnormPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* RmsnormPlugin::getPluginType() const noexcept
{
    return RMSNORM_PLUGIN_NAME;
}

const char* RmsnormPlugin::getPluginVersion() const noexcept
{
    return RMSNORM_PLUGIN_VERSION;
}

int RmsnormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int RmsnormPlugin::initialize() noexcept
{
    return 0;
}

void RmsnormPlugin::terminate() noexcept {}

size_t RmsnormPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mType);
}

void RmsnormPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void RmsnormPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

RmsnormPluginCreator::RmsnormPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RmsnormPluginCreator::getPluginName() const noexcept
{
    return RMSNORM_PLUGIN_NAME;
}

const char* RmsnormPluginCreator::getPluginVersion() const noexcept
{
    return RMSNORM_PLUGIN_VERSION;
}

const PluginFieldCollection* RmsnormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RmsnormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    float eps;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RmsnormPlugin(eps, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RmsnormPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RmsnormPlugin::destroy()
    try
    {
        auto* obj = new RmsnormPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
