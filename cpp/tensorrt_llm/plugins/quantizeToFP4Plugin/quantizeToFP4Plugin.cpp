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
#include "quantizeToFP4Plugin.h"
#include "pluginUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include <NvInferRuntimeBase.h>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::QuantizeToFP4PluginCreator;
using tensorrt_llm::plugins::QuantizeToFP4Plugin;

constexpr nvinfer1::DataType FP4_DTYPE = nvinfer1::DataType::kFP4;
constexpr nvinfer1::DataType FP8_DTYPE = nvinfer1::DataType::kFP8;

static char const* QUANT_FP4_PLUGIN_VERSION{"1"};
static char const* QUANT_FP4_PLUGIN_NAME{"QuantizeToFP4"};
PluginFieldCollection QuantizeToFP4PluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizeToFP4PluginCreator::mPluginAttributes;

QuantizeToFP4Plugin::QuantizeToFP4Plugin(){};

// Parameterized constructor
QuantizeToFP4Plugin::QuantizeToFP4Plugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QuantizeToFP4Plugin::clone() const noexcept
{
    auto* plugin = new QuantizeToFP4Plugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs QuantizeToFP4Plugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Quantized output in FP4 datatype.
    if (outputIndex == 0)
    {
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        // // Div up by 16 as the storage type has 16 FP4 values per element.
        // ret.d[ret.nbDims - 1]
        //     = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *ret.d[ret.nbDims - 1],
        //     *exprBuilder.constant(16));
        return ret;
    }
    // Scaling Factors in FP8.
    else if (outputIndex == 1)
    {
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        // Sequence dimension or token dimension.
        // Pad to multiple of 128.
        auto dimM
            = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *ret.d[ret.nbDims - 2], *exprBuilder.constant(128));
        ret.d[ret.nbDims - 2] = exprBuilder.operation(DimensionOperation::kPROD, *dimM, *exprBuilder.constant(128));
        // Hidden size dimension.
        // Div (rounding up) by 16 since 16 elements share one SF and SF padded to k%4==0.
        ret.d[ret.nbDims - 1]
            = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *ret.d[ret.nbDims - 1], *exprBuilder.constant(16));
        return ret;
    }
    return DimsExprs{};
}

bool QuantizeToFP4Plugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    // half input + float global_sf + fp4 output (e2m1) + fp8 SF output.
    int const totalPoses = 2 + 2;
    TLLM_CHECK(0 <= pos && pos < totalPoses);
    TLLM_CHECK(nbInputs == 2);
    switch (pos)
    {
    case 0:
        return (inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kBF16
                   || inOut[pos].type == nvinfer1::DataType::kFP8)
            && (inOut[pos].format == TensorFormat::kLINEAR);
    case 1: return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    case 2: return (inOut[pos].type == FP4_DTYPE) && (inOut[pos].format == TensorFormat::kLINEAR);
    case 3: return (inOut[pos].type == FP8_DTYPE) && (inOut[pos].format == TensorFormat::kLINEAR);
    default: break;
    }
    return false;
}

void QuantizeToFP4Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t QuantizeToFP4Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int QuantizeToFP4Plugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N] half data type
    //     SF scale [1] float data type
    //            used to scale SF from input range to fp8 range (448.f / (MaxVal of input / 6.f))
    // outputs
    //     output [M(*), N] fp4 storage (E2M1)
    //     SF output [M, N / 16] fp8 storage (UE4M3)

    int64_t m64 = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m64 *= inputDesc[0].dims.d[i];
    }
    int const m = TLLM_INT32_CAST(m64);
    int const n = TLLM_INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);

    TLLM_CHECK_WITH_INFO(n % 16 == 0, "the N dimension must be multiple of 16.");

    float const* SFScale = static_cast<float const*>(inputs[1]);
    int64_t* output = reinterpret_cast<int64_t*>(outputs[0]);
    int32_t* SFoutput = reinterpret_cast<int32_t*>(outputs[1]);

    DataType inputDtype = inputDesc[0].type;

    switch (inputDtype)
    {
    case DataType::kHALF:
    {
        auto input = reinterpret_cast<half const*>(inputs[0]);
        invokeFP4Quantization(1, m, n, input, SFScale, output, SFoutput, false, QuantizationSFLayout::SWIZZLED,
            mMultiProcessorCount, stream);
        break;
    }

    case DataType::kBF16:
    {
        auto input = reinterpret_cast<__nv_bfloat16 const*>(inputs[0]);
        invokeFP4Quantization(1, m, n, input, SFScale, output, SFoutput, false, QuantizationSFLayout::SWIZZLED,
            mMultiProcessorCount, stream);
        break;
    }

    case DataType::kFP8:
    {
        auto input = reinterpret_cast<__nv_fp8_e4m3 const*>(inputs[0]);
        invokeFP4Quantization(1, m, n, input, SFScale, output, SFoutput, false, QuantizationSFLayout::SWIZZLED,
            mMultiProcessorCount, stream);
        break;
    }

    default: TLLM_LOG_ERROR("only half, bfloat16 and fp8 data type are supported."); break;
    }

    // Use UE4M3 scales by default.
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType QuantizeToFP4Plugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        // Output 0 quantized output.
        return FP4_DTYPE;
    }
    // Output 1 SF (scaling factors).
    return FP8_DTYPE;
}

// IPluginV2 Methods

char const* QuantizeToFP4Plugin::getPluginType() const noexcept
{
    return QUANT_FP4_PLUGIN_NAME;
}

char const* QuantizeToFP4Plugin::getPluginVersion() const noexcept
{
    return QUANT_FP4_PLUGIN_VERSION;
}

int QuantizeToFP4Plugin::getNbOutputs() const noexcept
{
    return 2;
}

int QuantizeToFP4Plugin::initialize() noexcept
{
    return 0;
}

void QuantizeToFP4Plugin::terminate() noexcept {}

size_t QuantizeToFP4Plugin::getSerializationSize() const noexcept
{
    return 0;
}

void QuantizeToFP4Plugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    TLLM_CHECK(d == a + getSerializationSize());
}

void QuantizeToFP4Plugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

QuantizeToFP4PluginCreator::QuantizeToFP4PluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QuantizeToFP4PluginCreator::getPluginName() const noexcept
{
    return QUANT_FP4_PLUGIN_NAME;
}

char const* QuantizeToFP4PluginCreator::getPluginVersion() const noexcept
{
    return QUANT_FP4_PLUGIN_VERSION;
}

PluginFieldCollection const* QuantizeToFP4PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizeToFP4PluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* obj = new QuantizeToFP4Plugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizeToFP4PluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QuantizeToFP4Plugin::destroy()
    try
    {
        auto* obj = new QuantizeToFP4Plugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
