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
#include "rmsnormQuantizationPlugin.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::RmsnormQuantizationPluginCreator;
using tensorrt_llm::plugins::RmsnormQuantizationPlugin;

static const char* RMSNORM_QUANTIZATION_PLUGIN_VERSION{"1"};
static const char* RMSNORM_QUANTIZATION_PLUGIN_NAME{"RmsnormQuantization"};
PluginFieldCollection RmsnormQuantizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RmsnormQuantizationPluginCreator::mPluginAttributes;

RmsnormQuantizationPlugin::RmsnormQuantizationPlugin(float eps, bool dynamicActivationScaling, nvinfer1::DataType type)
    : mEps(eps)
    , mDynActScaling(dynamicActivationScaling)
    , mType(type)
{
}

// Parameterized constructor
RmsnormQuantizationPlugin::RmsnormQuantizationPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    read(d, mDynActScaling);
    read(d, mType);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RmsnormQuantizationPlugin::clone() const noexcept
{
    auto* plugin = new RmsnormQuantizationPlugin(mEps, mDynActScaling, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RmsnormQuantizationPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        // Quantized output
        return inputs[outputIndex];
    }

    // Dynamic scaling output if enabled
    try
    {
        TLLM_CHECK(outputIndex == 1);
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims - 1; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        ret.d[ret.nbDims - 1] = exprBuilder.constant(1);
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool RmsnormQuantizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const int totalPoses = 6 + static_cast<int>(mDynActScaling);
    TLLM_CHECK(0 <= pos && pos < totalPoses);
    TLLM_CHECK(nbInputs == 4);
    if (pos < nbInputs)
    {
        switch (pos)
        {
        case 0:
        case 1:
        case 2: return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        case 3: return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    if (pos == 4)
    {
        // Quantized output
        return (inOut[pos].type == nvinfer1::DataType::kINT8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    // Dynamic scaling if enabled
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RmsnormQuantizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RmsnormQuantizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RmsnormQuantizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     weight [N, ]
    //     bias [N, ]
    //     scale_to_int [1]
    // outputs
    //     output [M(*), N]
    //     dynamic_scaling [M(*), 1] (optional output)

    int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];

    const float* scale = reinterpret_cast<const float*>(inputs[3]);
    int8_t* output = reinterpret_cast<int8_t*>(outputs[0]);
    float* dynamic_scale = mDynActScaling ? reinterpret_cast<float*>(outputs[1]) : nullptr;

    if (mType == DataType::kHALF)
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        const half* bias = reinterpret_cast<const half*>(inputs[2]);
        invokeGeneralRmsNorm((half*) nullptr, input, weight, bias, mEps, m, n, stream, scale, dynamic_scale, output);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* input = reinterpret_cast<const float*>(inputs[0]);
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        const float* bias = reinterpret_cast<const float*>(inputs[2]);
        invokeGeneralRmsNorm((float*) nullptr, input, weight, bias, mEps, m, n, stream, scale, dynamic_scale, output);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RmsnormQuantizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert((mDynActScaling && index < 2) || (!mDynActScaling && index == 0));
    if (index == 0)
    {
        // Output 0 quantized output of layer norm
        return nvinfer1::DataType::kINT8;
    }
    // Output 1 dynamic act scaling
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods

const char* RmsnormQuantizationPlugin::getPluginType() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_NAME;
}

const char* RmsnormQuantizationPlugin::getPluginVersion() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_VERSION;
}

int RmsnormQuantizationPlugin::getNbOutputs() const noexcept
{
    return 1 + static_cast<int>(mDynActScaling);
}

int RmsnormQuantizationPlugin::initialize() noexcept
{
    return 0;
}

void RmsnormQuantizationPlugin::terminate() noexcept {}

size_t RmsnormQuantizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mDynActScaling) + sizeof(mType);
}

void RmsnormQuantizationPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mDynActScaling);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void RmsnormQuantizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

RmsnormQuantizationPluginCreator::RmsnormQuantizationPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("dyn_act_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RmsnormQuantizationPluginCreator::getPluginName() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_NAME;
}

const char* RmsnormQuantizationPluginCreator::getPluginVersion() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_VERSION;
}

const PluginFieldCollection* RmsnormQuantizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RmsnormQuantizationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    float eps;
    nvinfer1::DataType type;
    bool dynamicActivationScaling;
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
        else if (!strcmp(attrName, "dyn_act_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dynamicActivationScaling = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RmsnormQuantizationPlugin(eps, dynamicActivationScaling, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RmsnormQuantizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RmsnormQuantizationPlugin::destroy()
    try
    {
        auto* obj = new RmsnormQuantizationPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
