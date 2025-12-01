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
#include "quantizeTensorPlugin.h"
#include "tensorrt_llm/kernels/quantization.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::plugins::QuantizeTensorPluginCreator;
using tensorrt_llm::plugins::QuantizeTensorPlugin;

static char const* QUANTIZE_TENSOR_PLUGIN_VERSION{"1"};
static char const* QUANTIZE_TENSOR_PLUGIN_NAME{"QuantizeTensor"};
PluginFieldCollection QuantizeTensorPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizeTensorPluginCreator::mPluginAttributes;

QuantizeTensorPlugin::QuantizeTensorPlugin() {}

// Parameterized constructor
QuantizeTensorPlugin::QuantizeTensorPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QuantizeTensorPlugin::clone() const noexcept
{
    return new QuantizeTensorPlugin(*this);
}

nvinfer1::DimsExprs QuantizeTensorPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
        TLLM_CHECK(outputIndex < 1);
        // Quantized input
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QuantizeTensorPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF
#ifdef ENABLE_BF16
                   || inOut[pos].type == nvinfer1::DataType::kBF16
#endif
                   )
            && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // scales
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // quantized activation
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        TLLM_CHECK(false);
        return false;
    }
}

void QuantizeTensorPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t QuantizeTensorPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int QuantizeTensorPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     activation     [M(*), K]
    //     scale          [1, 1]
    // outputs
    //     quant          [M(*), K]

    int64_t numElts = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims; ++ii)
    {
        numElts *= inputDesc[0].dims.d[ii];
    }

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        invokeQuantization<float>(reinterpret_cast<int8_t*>(outputs[0]), reinterpret_cast<float const*>(inputs[0]),
            numElts, reinterpret_cast<float const*>(inputs[1]), stream, mProp.maxGridSize[0]);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        invokeQuantization<half>(reinterpret_cast<int8_t*>(outputs[0]), reinterpret_cast<half const*>(inputs[0]),
            numElts, reinterpret_cast<float const*>(inputs[1]), stream, mProp.maxGridSize[0]);
    }
#ifdef ENABLE_BF16
    else if (inputDesc[0].type == DataType::kBF16)
    {
        invokeQuantization<__nv_bfloat16>(reinterpret_cast<int8_t*>(outputs[0]),
            reinterpret_cast<__nv_bfloat16 const*>(inputs[0]), numElts, reinterpret_cast<float const*>(inputs[1]),
            stream, mProp.maxGridSize[0]);
    }
#endif
    sync_check_cuda_error(stream);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType QuantizeTensorPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(nbInputs == 2);
    TLLM_CHECK(index == 0);
    return nvinfer1::DataType::kINT8;
}

// IPluginV2 Methods

char const* QuantizeTensorPlugin::getPluginType() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_NAME;
}

char const* QuantizeTensorPlugin::getPluginVersion() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_VERSION;
}

int QuantizeTensorPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QuantizeTensorPlugin::initialize() noexcept
{
    int deviceId = 0;
    tensorrt_llm::common::check_cuda_error(cudaGetDevice(&deviceId));
    tensorrt_llm::common::check_cuda_error(cudaGetDeviceProperties(&mProp, deviceId));
    return 0;
}

void QuantizeTensorPlugin::terminate() noexcept {}

size_t QuantizeTensorPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void QuantizeTensorPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    TLLM_CHECK(d == a + getSerializationSize());
}

void QuantizeTensorPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

QuantizeTensorPluginCreator::QuantizeTensorPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QuantizeTensorPluginCreator::getPluginName() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_NAME;
}

char const* QuantizeTensorPluginCreator::getPluginVersion() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_VERSION;
}

PluginFieldCollection const* QuantizeTensorPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizeTensorPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* obj = new QuantizeTensorPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizeTensorPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QuantizeTensorPlugin::destroy()
    try
    {
        auto* obj = new QuantizeTensorPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
