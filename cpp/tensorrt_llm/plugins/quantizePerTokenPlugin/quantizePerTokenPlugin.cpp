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
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::plugins::QuantizePerTokenPluginCreator;
using tensorrt_llm::plugins::QuantizePerTokenPlugin;

static char const* QUANTIZE_PER_TOKEN_PLUGIN_VERSION{"1"};
static char const* QUANTIZE_PER_TOKEN_PLUGIN_NAME{"QuantizePerToken"};
PluginFieldCollection QuantizePerTokenPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizePerTokenPluginCreator::mPluginAttributes;

QuantizePerTokenPlugin::QuantizePerTokenPlugin(
    nvinfer1::DataType outputType, QuantMode quantMode, bool clampValEnabled, bool sumPerToken)
    : mOutputType{outputType}
    , mQuantMode{quantMode}
    , mClampValEnabled{clampValEnabled}
    , mSumPerToken{sumPerToken}
{
    TLLM_CHECK_WITH_INFO(mOutputType == nvinfer1::DataType::kINT8 || mOutputType == nvinfer1::DataType::kFP8,
        "Only int8 or fp8 output type is allowed.");
    // Check if the quant mode is valid.
    TLLM_CHECK_WITH_INFO(mQuantMode.hasPerTokenScaling(), "The quant mode is not valid.");
}

// Parameterized constructor
QuantizePerTokenPlugin::QuantizePerTokenPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mOutputType);
    read(d, mQuantMode);
    read(d, mClampValEnabled);
    read(d, mSumPerToken);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QuantizePerTokenPlugin::clone() const noexcept
{
    auto* plugin = new QuantizePerTokenPlugin(mOutputType, mQuantMode, mClampValEnabled, mSumPerToken);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs QuantizePerTokenPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs <= 2);
        TLLM_CHECK(outputIndex <= 2);
        if (outputIndex == 2)
        {
            // Per token sums.
            TLLM_CHECK(mSumPerToken);
        }

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
        // [M(*), 1] dynamic per token scales or sums
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QuantizePerTokenPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == 0)
    {
        // activation
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF
#ifdef ENABLE_BF16
                   || inOut[pos].type == nvinfer1::DataType::kBF16
#endif
                   )
            && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == 1 && mClampValEnabled)
    {
        // clamp_max_v
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == 1 + int(mClampValEnabled))
    {
        // quantized activation
        return inOut[pos].type == mOutputType && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == 2 + int(mClampValEnabled))
    {
        // scales
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == 3 + int(mClampValEnabled))
    {
        TLLM_CHECK(mSumPerToken);
        // per-token sums
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    // Never should be here
    assert(false);
    return false;
}

void QuantizePerTokenPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t QuantizePerTokenPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

template <typename T, typename QuantT>
void QuantizePerTokenPlugin::dispatchDataType(void* output, void const* input, void const* clampValPtr, void* scalePtr,
    void* sumPtr, int dim0, int dim1, cudaStream_t stream) noexcept
{
    // inputs
    //     activation     [dim0(*), dim1]
    //     clamp_value    [2], contains min val, and max val (optional)
    // outputs
    //     quant          [dim0(*), dim1]
    //     scale_tokens   [dim0(*), 1]

    invokePerTokenQuantization(reinterpret_cast<QuantT*>(output), reinterpret_cast<T const*>(input), dim0, dim1,
        reinterpret_cast<float const*>(clampValPtr), reinterpret_cast<float*>(scalePtr),
        reinterpret_cast<float*>(sumPtr), mQuantMode, stream);
}

int QuantizePerTokenPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     activation     [M(*), K]
    //     clamp_value    [2], contains min val, and max val (optional)
    // outputs
    //     quant          [M(*), K]     Quantized activations.
    //     scale_tokens   [M(*), 1]     Per-token scales.
    //     token_sums     [M(*), 1]     (Optional) Per-token sums of all the channels (before quantization).

    int64_t m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    int64_t const k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    void const* clampValPtr = mClampValEnabled ? inputs[1] : nullptr;
    void* sumPtr = mSumPerToken ? outputs[2] : nullptr;

    if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kINT8)
    {
        dispatchDataType<float, int8_t>(outputs[0], inputs[0], clampValPtr, outputs[1], sumPtr, m, k, stream);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kFP8)
    {
        dispatchDataType<float, __nv_fp8_e4m3>(outputs[0], inputs[0], clampValPtr, outputs[1], sumPtr, m, k, stream);
    }
#endif // ENABLE_FP8
    else if (inputDesc[0].type == DataType::kHALF && mOutputType == DataType::kINT8)
    {
        dispatchDataType<half, int8_t>(outputs[0], inputs[0], clampValPtr, outputs[1], sumPtr, m, k, stream);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kHALF && mOutputType == DataType::kFP8)
    {
        dispatchDataType<half, __nv_fp8_e4m3>(outputs[0], inputs[0], clampValPtr, outputs[1], sumPtr, m, k, stream);
    }
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
    else if (inputDesc[0].type == DataType::kBF16 && mOutputType == DataType::kINT8)
    {
        dispatchDataType<__nv_bfloat16, int8_t>(outputs[0], inputs[0], clampValPtr, outputs[1], sumPtr, m, k, stream);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kBF16 && mOutputType == DataType::kFP8)
    {
        dispatchDataType<__nv_bfloat16, __nv_fp8_e4m3>(
            outputs[0], inputs[0], clampValPtr, outputs[1], sumPtr, m, k, stream);
    }
#endif // ENABLE_FP8
#endif // ENABLE_BF16
    sync_check_cuda_error(stream);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType QuantizePerTokenPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(nbInputs >= 1);
    TLLM_CHECK(index <= 2);
    if (index == 2)
    {
        // Per token sums.
        TLLM_CHECK(mSumPerToken);
    }
    return index == 0 ? mOutputType : nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods

char const* QuantizePerTokenPlugin::getPluginType() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_NAME;
}

char const* QuantizePerTokenPlugin::getPluginVersion() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_VERSION;
}

int QuantizePerTokenPlugin::getNbOutputs() const noexcept
{
    return 2 + static_cast<int>(mSumPerToken);
}

int QuantizePerTokenPlugin::initialize() noexcept
{
    return 0;
}

void QuantizePerTokenPlugin::terminate() noexcept {}

size_t QuantizePerTokenPlugin::getSerializationSize() const noexcept
{
    return sizeof(mOutputType) + sizeof(mQuantMode) + sizeof(mClampValEnabled) + sizeof(mSumPerToken);
}

void QuantizePerTokenPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mOutputType);
    write(d, mQuantMode);
    write(d, mClampValEnabled);
    write(d, mSumPerToken);
    TLLM_CHECK(d == a + getSerializationSize());
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
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("quant_mode", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("clamp_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("sum_per_token", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QuantizePerTokenPluginCreator::getPluginName() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_NAME;
}

char const* QuantizePerTokenPluginCreator::getPluginVersion() const noexcept
{
    return QUANTIZE_PER_TOKEN_PLUGIN_VERSION;
}

PluginFieldCollection const* QuantizePerTokenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizePerTokenPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginFieldParser p{fc->nbFields, fc->fields};
    try
    {
        auto* obj = new QuantizePerTokenPlugin(static_cast<nvinfer1::DataType>(p.getScalar<int32_t>("type_id").value()),
            QuantMode(p.getScalar<int32_t>("quant_mode").value()),
            static_cast<bool>(p.getScalar<int8_t>("clamp_enabled").value()),
            static_cast<bool>(p.getScalar<int32_t>("sum_per_token").value()));

        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizePerTokenPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QuantizePerTokenPlugin::destroy()
    try
    {
        auto* obj = new QuantizePerTokenPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
